"""
배드민턴 서브 폴트 검출기 v2
- 선수 신체 비율 자동 계산 (픽셀 절대값 X → 신체 비율 기준)
- 손목 속도 기반 임팩트 자동 감지
- 임팩트 전 구간(pre-impact window) 폴트 분석
- 서브 25개 자동 분리
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ── COCO 17 keypoint 인덱스 ────────────────────────────────
KP = {
    "nose": 0,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}
CONF_THR = 0.4


def get_kp(kps, name):
    idx = KP[name]
    x, y, c = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x, y) if c >= CONF_THR else None


# ── 신체 비율 계산 ─────────────────────────────────────────
def body_metrics(kps):
    """
    BWF 규정 9.1.6 기준:
    허리선 = 갈비뼈 최하단 (어깨~엉덩이 사이 약 60% 지점)
    COCO에 갈비뼈 키포인트가 없으므로 어깨-엉덩이 비율로 추정
    """
    ls = get_kp(kps, "left_shoulder")
    rs = get_kp(kps, "right_shoulder")
    lh = get_kp(kps, "left_hip")
    rh = get_kp(kps, "right_hip")
    la = get_kp(kps, "left_ankle")
    ra = get_kp(kps, "right_ankle")

    shoulder_y = None
    if ls and rs:
        shoulder_y = (ls[1] + rs[1]) / 2
    elif ls:
        shoulder_y = ls[1]
    elif rs:
        shoulder_y = rs[1]

    hip_y = None
    if lh and rh:
        hip_y = (lh[1] + rh[1]) / 2
    elif lh:
        hip_y = lh[1]
    elif rh:
        hip_y = rh[1]

    ankle_y = None
    if la and ra:
        ankle_y = (la[1] + ra[1]) / 2
    elif la:
        ankle_y = la[1]
    elif ra:
        ankle_y = ra[1]

    body_height = None
    if shoulder_y and ankle_y:
        body_height = ankle_y - shoulder_y

    # BWF 9.1.6: 허리 = 갈비뼈 최하단 = 어깨에서 엉덩이까지의 60% 지점
    # 엉덩이(hip)보다 위에 위치 → 더 엄격한 기준
    waist_y = None
    if shoulder_y and hip_y:
        waist_y = shoulder_y + (hip_y - shoulder_y) * 0.60

    return {
        "shoulder_y": shoulder_y,
        "hip_y": hip_y,
        "waist_y": waist_y,   # 갈비뼈 최하단 추정 (BWF 기준)
        "ankle_y": ankle_y,
        "body_height": body_height,
        "valid": all([shoulder_y, hip_y, waist_y, ankle_y,
                      body_height and body_height > 50]),
    }


# ── 폴트 판정 (신체 비율 기반) ────────────────────────────
def detect_faults(kps, side="right"):
    """
    BWF 9.1.6 웨이스트 폴트:
      임팩트 순간 셔틀콕(= 손목 위치로 근사)이 갈비뼈 최하단보다 위
    BWF 9.1.7 샤프트 폴트:
      임팩트 순간 라켓 샤프트+헤드가 아래를 향해야 함
      → 손목(그립 끝)이 팔꿈치보다 아래에 있어야 정상
      → 팔꿈치가 손목보다 아래(y 큼)면 라켓이 위를 향함 = 폴트
    마진: 신체 높이의 2% (오차 흡수)
    """
    wrist   = get_kp(kps, f"{side}_wrist")
    elbow   = get_kp(kps, f"{side}_elbow")
    metrics = body_metrics(kps)

    result = {
        "waist_fault": False,
        "shaft_fault": False,
        "waist_margin": None,   # 양수 = 폴트 깊이(px), 음수 = 정상 여유
        "details": {},
    }

    if not metrics["valid"]:
        return result

    margin = metrics["body_height"] * 0.02

    # 웨이스트 폴트: 손목 y < 갈비뼈 y (손목이 허리 위)
    if wrist and metrics["waist_y"]:
        diff = metrics["waist_y"] - wrist[1]   # 양수 = 손목이 허리 위
        result["waist_margin"] = round(diff, 1)
        if diff > margin:
            result["waist_fault"] = True
        result["details"]["wrist_y"] = round(wrist[1], 1)
        result["details"]["waist_y"] = round(metrics["waist_y"], 1)
        result["details"]["hip_y"]   = round(metrics["hip_y"], 1)
        result["details"]["body_h"]  = round(metrics["body_height"], 1)

    # 샤프트 폴트: 팔꿈치 y > 손목 y (팔꿈치가 손목보다 아래 → 라켓 위를 향함)
    # 정상 서브: 팔꿈치가 손목보다 위(y 작음), 라켓이 아래를 향함
    if wrist and elbow:
        shaft_diff = elbow[1] - wrist[1]   # 양수 = 팔꿈치가 손목보다 아래 = 폴트
        if shaft_diff > margin:
            result["shaft_fault"] = True
        result["details"]["elbow_y"]    = round(elbow[1], 1)
        result["details"]["shaft_diff"] = round(shaft_diff, 1)

    return result


# ── 임팩트 자동 감지 ───────────────────────────────────────
def find_impact_frames(wrist_positions, fps, min_gap_sec=2.0):
    """
    손목 속도 피크 = 임팩트 후보
    min_gap_sec: 같은 서브로 묶지 않을 최소 간격
    """
    if len(wrist_positions) < 3:
        return []

    frames = sorted(wrist_positions.keys())
    speeds = {}
    for i in range(1, len(frames) - 1):
        f = frames[i]
        prev_f = frames[i - 1]
        next_f = frames[i + 1]
        if wrist_positions[f] and wrist_positions[prev_f]:
            dx = wrist_positions[f][0] - wrist_positions[prev_f][0]
            dy = wrist_positions[f][1] - wrist_positions[prev_f][1]
            speeds[f] = (dx**2 + dy**2) ** 0.5
        else:
            speeds[f] = 0

    if not speeds:
        return []

    speed_vals = list(speeds.values())
    threshold = np.mean(speed_vals) + 1.5 * np.std(speed_vals)

    min_gap_frames = int(min_gap_sec * fps)
    impacts = []
    last_impact = -min_gap_frames

    for f in frames:
        if f not in speeds:
            continue
        if speeds[f] >= threshold and (f - last_impact) >= min_gap_frames:
            impacts.append(f)
            last_impact = f

    return impacts


# ── 서브 구간 분석 ─────────────────────────────────────────
def analyze_serve(frames_data, impact_frame, pre_window=15, side="right"):
    """
    임팩트 전 pre_window 프레임 동안 폴트 여부 집계.
    frames_data: {frame_idx: kps_array}
    """
    start = max(0, impact_frame - pre_window)
    window_frames = range(start, impact_frame + 1)

    waist_faults = []
    shaft_faults = []
    max_waist_margin = -9999

    for f in window_frames:
        if f not in frames_data:
            continue
        kps = frames_data[f]
        fault = detect_faults(kps, side)
        if fault["waist_fault"]:
            waist_faults.append(f)
        if fault["shaft_fault"]:
            shaft_faults.append(f)
        if fault["waist_margin"] is not None:
            max_waist_margin = max(max_waist_margin, fault["waist_margin"])

    fault_ratio = len(waist_faults) / max(len(list(window_frames)), 1)

    return {
        "impact_frame": impact_frame,
        "impact_time_sec": None,  # 나중에 채움
        "waist_fault": fault_ratio >= 0.4,      # 40% 이상 프레임에서 폴트
        "shaft_fault": len(shaft_faults) >= 3,
        "waist_fault_frames": len(waist_faults),
        "shaft_fault_frames": len(shaft_faults),
        "max_waist_margin_px": round(max_waist_margin, 1) if max_waist_margin != -9999 else None,
        "window_frames": pre_window,
    }


# ── 결과 영상 드로잉 ───────────────────────────────────────
def draw_overlay(frame, kps, fault, metrics, frame_idx, fps, is_impact=False):
    h, w = frame.shape[:2]

    # 갈비뼈 최하단 = BWF 허리 기준선 (노란색)
    if metrics["waist_y"]:
        wy = int(metrics["waist_y"])
        cv2.line(frame, (0, wy), (w, wy), (0, 220, 220), 2)
        cv2.putText(frame, "WAIST (BWF 9.1.6)", (8, wy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
    # 엉덩이선 참고용 (회색)
    if metrics.get("hip_y"):
        hy = int(metrics["hip_y"])
        cv2.line(frame, (0, hy), (w, hy), (120, 120, 120), 1)
        cv2.putText(frame, "HIP", (8, hy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # 임팩트 표시
    if is_impact:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 4)
        cv2.putText(frame, "IMPACT", (w // 2 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # 폴트 알림
    y = 55
    if fault["waist_fault"]:
        cv2.putText(frame, "FAULT: WAIST",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 32
    if fault["shaft_fault"]:
        cv2.putText(frame, "FAULT: SHAFT",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        y += 32
    if not fault["waist_fault"] and not fault["shaft_fault"]:
        cv2.putText(frame, "OK",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 60), 2)

    # 타임스탬프
    ts = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"{ts:.2f}s  f={frame_idx}",
                (w - 200, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)
    return frame


# ── 메인 ──────────────────────────────────────────────────
def analyze_video(input_path, output_path=None, side="right",
                  pre_window=15, expected_serves=None):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"파일 없음: {input_path}")
    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_result{input_path.suffix}")

    model = YOLO("yolov8n-pose.pt")
    cap   = cv2.VideoCapture(str(input_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[1/3] 포즈 추출 중... ({total} frames)")

    # 1패스: 전 프레임 포즈 추출
    frames_data    = {}   # {frame_idx: kps}
    wrist_positions = {}  # {frame_idx: (x, y)}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps = results[0].keypoints.data[0].cpu().numpy()
            frames_data[frame_idx] = kps
            w = get_kp(kps, f"{side}_wrist")
            wrist_positions[frame_idx] = w
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()

    # 2패스: 임팩트 감지
    print(f"\n[2/3] 임팩트 감지 중...")
    impacts = find_impact_frames(wrist_positions, fps, min_gap_sec=2.0)
    print(f"  감지된 임팩트: {len(impacts)}개 (예상: {expected_serves}개)")

    # 예상 서브 수와 차이 경고
    if expected_serves and abs(len(impacts) - expected_serves) > 3:
        print(f"  ⚠️  감지 수가 예상과 차이납니다. --pre_window 조정을 시도해보세요.")

    # 서브별 폴트 분석
    serves = []
    for imp in impacts:
        result = analyze_serve(frames_data, imp, pre_window, side)
        result["impact_time_sec"] = round(imp / fps, 2)
        serves.append(result)

    # 3패스: 결과 영상 생성
    print(f"\n[3/3] 결과 영상 생성 중...")
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    impact_set = set(impacts)
    # 임팩트 전후 하이라이트 구간
    highlight_range = set()
    for imp in impacts:
        for f in range(max(0, imp - pre_window), imp + 5):
            highlight_range.add(f)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_data:
            kps     = frames_data[frame_idx]
            metrics = body_metrics(kps)
            fault   = detect_faults(kps, side)
            is_imp  = frame_idx in impact_set

            results = model(frame, verbose=False)
            annotated = results[0].plot(boxes=False)
            draw_overlay(annotated, kps, fault, metrics, frame_idx, fps, is_imp)
            out.write(annotated)
        else:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    # 리포트
    fault_serves = [s for s in serves if s["waist_fault"] or s["shaft_fault"]]
    report = {
        "input": str(input_path),
        "fps": fps,
        "total_frames": total,
        "side": side,
        "pre_impact_window_frames": pre_window,
        "total_serves_detected": len(serves),
        "fault_serves": len(fault_serves),
        "ok_serves": len(serves) - len(fault_serves),
        "serves": serves,
    }

    report_path = str(Path(output_path).with_suffix(".json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"완료!")
    print(f"  결과 영상: {output_path}")
    print(f"  리포트:   {report_path}")
    print(f"  총 서브:  {len(serves)}개 감지")
    print(f"  폴트:     {len(fault_serves)}개")
    print(f"  정상:     {len(serves) - len(fault_serves)}개")
    print()
    for i, s in enumerate(serves):
        status = []
        if s["waist_fault"]: status.append("웨이스트폴트")
        if s["shaft_fault"]:  status.append("샤프트폴트")
        label = " / ".join(status) if status else "정상"
        print(f"  서브 {i+1:02d}  {s['impact_time_sec']:6.2f}s  →  {label}")

    return report


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="배드민턴 서브 폴트 검출 v2")
    parser.add_argument("input")
    parser.add_argument("-o", "--output",   default=None)
    parser.add_argument("-s", "--side",     default="right", choices=["right", "left"])
    parser.add_argument("-w", "--pre_window", type=int, default=15,
                        help="임팩트 전 분석 프레임 수 (default: 15 = 0.5초)")
    parser.add_argument("-n", "--serves",   type=int, default=None,
                        help="예상 서브 개수 (감지 결과 검증용)")
    args = parser.parse_args()

    analyze_video(args.input, args.output, args.side, args.pre_window, args.serves)
