"""
국제시합 서브 높이 폴트 감지기
- 판정: 1.15m 높이 폴트만
- 감지: 정지/움직임 셔틀콕 + 사람 포즈 스켈레톤
- UI: 1.15m 선 + 포즈 스켈레톤 + 우측 하단 결과창
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

_KO_FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
_ko_fonts = {}

def _ko_font(size):
    if size not in _ko_fonts:
        try:
            _ko_fonts[size] = ImageFont.truetype(_KO_FONT_PATH, size)
        except Exception:
            _ko_fonts[size] = ImageFont.load_default()
    return _ko_fonts[size]

def put_ko(frame, text, pos, size=28, color=(255, 255, 255)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=_ko_font(size), fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# COCO 17 keypoint 이름 → 인덱스
KP_IDX = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

# 스켈레톤 연결선 (인덱스 쌍)
SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SHUTTLE_CLASS = 0
CONF_KP = 0.4  # 키포인트 신뢰도 임계값


def get_kp(kps, name):
    idx = KP_IDX.get(name)
    if idx is None or idx >= len(kps):
        return None
    x, y, conf = kps[idx]
    return (float(x), float(y)) if conf > CONF_KP else None


def calc_height_thresh(kps, player_height_m=1.70):
    la = get_kp(kps, "left_ankle")
    ra = get_kp(kps, "right_ankle")
    ls = get_kp(kps, "left_shoulder")
    rs = get_kp(kps, "right_shoulder")
    ankle_y    = np.mean([p[1] for p in [la, ra] if p]) if (la or ra) else None
    shoulder_y = np.mean([p[1] for p in [ls, rs] if p]) if (ls or rs) else None
    if ankle_y is None or shoulder_y is None:
        return None
    body_px = ankle_y - shoulder_y
    if body_px < 50:
        return None
    px_per_m = body_px / (player_height_m * 0.90)
    return ankle_y - (1.15 * px_per_m)


def calibrate(frames_kps, calibration_frames=90, player_height_m=1.70):
    thresholds = []
    for f in sorted(frames_kps)[:calibration_frames]:
        t = calc_height_thresh(frames_kps[f], player_height_m)
        if t is not None:
            thresholds.append(t)
    return float(np.median(thresholds)) if thresholds else None


def detect_shuttle(det_model, frame, conf_thr=0.10):
    results = det_model(frame, verbose=False)
    best = None
    if results[0].boxes is None:
        return None
    for box in results[0].boxes:
        if int(box.cls[0]) != SHUTTLE_CLASS:
            continue
        conf = float(box.conf[0])
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        if best is None or conf > best[2]:
            best = ((x1+x2)/2, (y1+y2)/2, conf, x1, y1, x2, y2)
    return best


def is_stationary(positions, frame_idx, window=8, max_disp=18):
    recent = [
        positions[f] for f in range(frame_idx - window, frame_idx + 1)
        if f in positions and positions[f] is not None
    ]
    if len(recent) < window // 2:
        return False
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    return (max(xs) - min(xs)) < max_disp and (max(ys) - min(ys)) < max_disp


def draw_skeleton(frame, kps):
    """포즈 스켈레톤 + 키포인트 그리기"""
    if kps is None:
        return
    # 연결선
    for i, j in SKELETON:
        if i >= len(kps) or j >= len(kps):
            continue
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci < CONF_KP or cj < CONF_KP:
            continue
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 230, 255), 2)
    # 키포인트 점
    for x, y, conf in kps:
        if conf < CONF_KP:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)


def draw_result_box(frame, fault_state, serves_so_far):
    """우측 하단 결과창"""
    H, W = frame.shape[:2]
    box_w, box_h = 220, 110
    x0, y0 = W - box_w - 16, H - box_h - 16

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1)

    fault_cnt = sum(1 for s in serves_so_far if s["height_fault"])
    ok_cnt    = len(serves_so_far) - fault_cnt

    put_ko(frame, "1.15m 판정", (x0 + 10, y0 + 6),  size=18, color=(200, 200, 200))
    put_ko(frame, f"폴트  {fault_cnt}",  (x0 + 10, y0 + 32), size=22, color=(80, 80, 255))
    put_ko(frame, f"정상  {ok_cnt}",    (x0 + 10, y0 + 58), size=22, color=(80, 255, 80))

    # 현재 판정 강조
    if fault_state is True:
        color = (0, 0, 255)
        label = "▶ FAULT"
    elif fault_state is False:
        color = (0, 200, 60)
        label = "▶ OK"
    else:
        color = (120, 120, 120)
        label = ""

    if label:
        put_ko(frame, label, (x0 + 10, y0 + 82), size=22, color=color)


def draw_frame(frame, shuttle, kps, height_thresh_y, fault_state, serves_so_far, frame_idx, fps):
    H, W = frame.shape[:2]

    # 포즈 스켈레톤
    draw_skeleton(frame, kps)

    # 1.15m 기준선
    if height_thresh_y is not None:
        hy = int(height_thresh_y)
        cv2.line(frame, (0, hy), (W, hy), (0, 200, 255), 2)
        put_ko(frame, "1.15m", (8, hy - 30), size=22, color=(0, 200, 255))

    # 셔틀콕 표시
    if shuttle:
        sx, sy   = int(shuttle[0]), int(shuttle[1])
        sy1      = int(shuttle[4])
        is_above = height_thresh_y is not None and sy1 < height_thresh_y
        color    = (0, 60, 255) if is_above else (0, 255, 80)
        cv2.circle(frame, (sx, sy), 16, color, 2)
        cv2.circle(frame, (sx, sy1), 5, color, -1)

    # 우측 하단 결과창
    draw_result_box(frame, fault_state, serves_so_far)

    # 타임스탬프
    ts = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"{ts:.2f}s", (W - 100, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def analyze(input_path, output_path, det_model_path,
            player_height_m=1.70, calibration_frames=90,
            result_display_sec=3.0, conf_thr=0.20):

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"파일 없음: {input_path}")
    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_intl_result.mp4")

    pose_model = YOLO("yolov8n-pose.pt")
    det_model  = YOLO(det_model_path)
    print(f"셔틀콕 모델: {det_model_path}")

    cap   = cv2.VideoCapture(str(input_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[1/2] 프레임 분석 중... ({total}프레임, {fps:.0f}fps)")

    frames_kps  = {}   # frame_idx → kps array (17×3)
    shuttle_pos = {}   # frame_idx → (cx, cy) or None
    shuttle_det = {}   # frame_idx → full tuple or None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_res = pose_model(frame, verbose=False)
        if pose_res[0].keypoints is not None and len(pose_res[0].keypoints) > 0:
            kps = pose_res[0].keypoints.data[0].cpu().numpy()
            frames_kps[frame_idx] = kps

        sh = detect_shuttle(det_model, frame, conf_thr)
        shuttle_det[frame_idx] = sh
        shuttle_pos[frame_idx] = (sh[0], sh[1]) if sh else None

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")

    cap.release()

    height_thresh_y = calibrate(frames_kps, calibration_frames, player_height_m)
    print(f"\n1.15m 기준선: {height_thresh_y:.0f}px" if height_thresh_y else "\n1.15m 기준선 감지 실패")

    print(f"\n[2/2] 결과 영상 생성 중...")
    cap    = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    result_display_f = int(result_display_sec * fps)
    fault_active     = None  # (is_fault, expire_frame)
    serves           = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sh  = shuttle_det.get(frame_idx)
        kps = frames_kps.get(frame_idx)

        # 정지 셔틀콕 + 1.15m 판정
        if is_stationary(shuttle_pos, frame_idx) and sh is not None and height_thresh_y is not None:
            shuttle_top_y  = sh[4]
            nearest        = min(frames_kps.keys(), key=lambda f: abs(f - frame_idx)) if frames_kps else None
            body_h_approx  = 200.0
            if nearest is not None:
                nkps       = frames_kps[nearest]
                la = get_kp(nkps, "left_ankle");  ra = get_kp(nkps, "right_ankle")
                ls = get_kp(nkps, "left_shoulder"); rs = get_kp(nkps, "right_shoulder")
                ankle_y    = np.mean([p[1] for p in [la, ra] if p]) if (la or ra) else None
                shoulder_y = np.mean([p[1] for p in [ls, rs] if p]) if (ls or rs) else None
                if ankle_y and shoulder_y:
                    body_h_approx = ankle_y - shoulder_y
            margin   = body_h_approx * 0.02
            is_fault = bool((height_thresh_y - shuttle_top_y) > margin)

            if fault_active is None or frame_idx > fault_active[1]:
                serves.append({
                    "frame":           frame_idx,
                    "time_sec":        round(frame_idx / fps, 2),
                    "height_fault":    is_fault,
                    "shuttle_top_y":   round(shuttle_top_y, 1),
                    "height_thresh_y": round(height_thresh_y, 1),
                })
                fault_active = (is_fault, frame_idx + result_display_f)

        current_fault = None
        if fault_active is not None:
            if frame_idx <= fault_active[1]:
                current_fault = fault_active[0]
            else:
                fault_active = None

        draw_frame(frame, sh, kps, height_thresh_y, current_fault, serves, frame_idx, fps)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    fault_count = sum(1 for s in serves if s["height_fault"])
    print(f"\n완료: {output_path}")
    print(f"판정: 총 {len(serves)}개 | 폴트 {fault_count}개 | 정상 {len(serves)-fault_count}개")

    json_path = str(Path(output_path).with_suffix(".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "input":            str(input_path),
            "fps":              fps,
            "det_model":        det_model_path,
            "height_thresh_px": height_thresh_y,
            "player_height_m":  player_height_m,
            "total_events":     len(serves),
            "fault_count":      fault_count,
            "events":           serves,
        }, f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_path}")
    return serves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="국제시합 1.15m 높이 폴트 감지기")
    parser.add_argument("input",                                   help="입력 영상")
    parser.add_argument("-o", "--output",       default=None,      help="출력 영상 경로")
    parser.add_argument("--det_model",          default="best_v9.pt", help="셔틀콕 YOLO 모델")
    parser.add_argument("--player_height",      type=float, default=1.70)
    parser.add_argument("--calibration_frames", type=int,   default=90)
    parser.add_argument("--result_display_sec", type=float, default=3.0)
    parser.add_argument("--conf",               type=float, default=0.10, help="셔틀콕 감지 신뢰도")
    args = parser.parse_args()

    analyze(
        args.input, args.output, args.det_model,
        player_height_m=args.player_height,
        calibration_frames=args.calibration_frames,
        result_display_sec=args.result_display_sec,
        conf_thr=args.conf,
    )
