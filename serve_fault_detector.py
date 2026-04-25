"""
배드민턴 서브 폴트 검출기 v5
주요 변경:
- 샤프트 폴트: 라켓헤드 없으면 감지 안 함 (팔꿈치 근사 제거 — 부정확)
- 결과 표시: 서브 완료(임팩트) 후 result_display_sec 초 동안 크게 표시
- 그 외 프레임: 기준선(허리/1.10m/서비스라인)만 표시
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
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

SKELETON_CONNECTIONS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6),
]

SHUTTLE_CLASS = 0
RACKET_CLASS  = 1


def get_kp(kps, name):
    idx = KP[name]
    x, y, c = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x, y) if c >= CONF_THR else None


def draw_skeleton(frame, kps):
    for i, j in SKELETON_CONNECTIONS:
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci >= CONF_THR and cj >= CONF_THR:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 255, 128), 2)
    for idx in range(len(kps)):
        x, y, c = kps[idx]
        if c >= CONF_THR:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 200, 255), -1)


# ── 신체 비율 계산 ─────────────────────────────────────────
def body_metrics(kps):
    ls = get_kp(kps, "left_shoulder");  rs = get_kp(kps, "right_shoulder")
    lh = get_kp(kps, "left_hip");       rh = get_kp(kps, "right_hip")
    la = get_kp(kps, "left_ankle");     ra = get_kp(kps, "right_ankle")

    def avg_y(a, b):
        if a and b: return (a[1] + b[1]) / 2
        return (a or b or [None, None])[1]

    shoulder_y = avg_y(ls, rs)
    hip_y      = avg_y(lh, rh)
    ankle_y    = avg_y(la, ra)

    body_height = (ankle_y - shoulder_y) if (shoulder_y and ankle_y) else None
    waist_y     = (shoulder_y + (hip_y - shoulder_y) * 0.60) if (shoulder_y and hip_y) else None

    return {
        "shoulder_y":  shoulder_y,
        "hip_y":       hip_y,
        "waist_y":     waist_y,
        "ankle_y":     ankle_y,
        "body_height": body_height,
        "valid": all([shoulder_y, hip_y, waist_y, ankle_y,
                      body_height and body_height > 50]),
    }


def _height_thresh_from_kps(kps, player_height_m):
    nose    = get_kp(kps, "nose")
    metrics = body_metrics(kps)
    if not metrics["valid"] or not nose:
        return None
    body_px = metrics["ankle_y"] - nose[1]
    if body_px <= 50:
        return None
    px_per_m = body_px / (player_height_m * 0.90)
    return metrics["ankle_y"] - (1.10 * px_per_m)


# ── 신체 기준선 1회 보정 ───────────────────────────────────
def calibrate_body(frames_data, calibration_frames=90, player_height_m=1.70):
    waist_ys, height_ys, body_hs = [], [], []
    ankle_ys, ankle_xs = [], []

    for f in sorted(frames_data.keys())[:calibration_frames]:
        kps = frames_data[f]
        m   = body_metrics(kps)
        if m["valid"]:
            waist_ys.append(m["waist_y"])
            body_hs.append(m["body_height"])
        h = _height_thresh_from_kps(kps, player_height_m)
        if h:
            height_ys.append(h)
        la = get_kp(kps, "left_ankle")
        ra = get_kp(kps, "right_ankle")
        for p in [la, ra]:
            if p:
                ankle_ys.append(p[1])
                ankle_xs.append(p[0])

    return {
        "waist_y":         float(np.median(waist_ys))  if waist_ys  else None,
        "height_thresh_y": float(np.median(height_ys)) if height_ys else None,
        "ankle_y_ref":     float(np.median(ankle_ys))  if ankle_ys  else None,
        "ankle_x_ref":     float(np.median(ankle_xs))  if ankle_xs  else None,
        "body_height":     float(np.median(body_hs))   if body_hs   else 200.0,
    }


# ── 커스텀 YOLO 검출 ───────────────────────────────────────
def detect_objects(det_model, frame, conf_thr=0.3):
    results = det_model(frame, verbose=False)
    shuttle = racket = None
    if results[0].boxes is None:
        return shuttle, racket
    for box in results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cls == SHUTTLE_CLASS and (shuttle is None or conf > shuttle[2]):
            shuttle = (cx, cy, conf, x1, y1, x2, y2)
        elif cls == RACKET_CLASS and (racket is None or conf > racket[2]):
            racket  = (cx, cy, conf, x1, y1, x2, y2)
    return shuttle, racket


# ── 폴트 판정 ─────────────────────────────────────────────
def detect_faults(kps, side="right", shuttle=None, racket=None, calib=None):
    """
    샤프트 폴트: 라켓헤드 감지된 경우에만 판정 (팔꿈치 근사 제거).
    BWF 9.1.7: 타격 순간 샤프트+헤드가 아래를 향해야 함 → 헤드 y > 손목 y
    """
    wrist  = get_kp(kps, f"{side}_wrist")
    m      = body_metrics(kps)

    waist_y_ref  = (calib or {}).get("waist_y")  or m.get("waist_y")
    h_thresh_ref = (calib or {}).get("height_thresh_y")
    body_h       = (calib or {}).get("body_height") or m.get("body_height") or 200.0
    margin       = body_h * 0.02

    result = {
        "waist_fault":   False,
        "height_fault":  False,
        "shaft_fault":   False,
        "waist_margin":  None,
        "height_margin": None,
        "shuttle_used":  shuttle is not None,
        "racket_used":   racket is not None,
        "details":       {},
    }

    # ── 웨이스트 폴트 (9.1.6) ─────────────────────────────
    # 셔틀콕 위치 우선, 없으면 손목 근사
    impact_y = shuttle[1] if shuttle else (wrist[1] if wrist else None)
    if impact_y is not None and waist_y_ref:
        diff = waist_y_ref - impact_y
        result["waist_margin"] = round(diff, 1)
        result["waist_fault"]  = diff > margin
        result["details"]["impact_y"] = round(impact_y, 1)
        result["details"]["waist_y"]  = round(waist_y_ref, 1)

    # ── 높이 폴트 (9.1.6.2) — 셔틀콕 있을 때만 ────────────
    if shuttle and h_thresh_ref:
        h_diff = h_thresh_ref - shuttle[1]
        result["height_margin"] = round(h_diff, 1)
        result["height_fault"]  = h_diff > margin
        result["details"]["height_thresh_y"] = round(h_thresh_ref, 1)

    # ── 샤프트 폴트 (9.1.7) — 라켓헤드 있을 때만 ──────────
    # 헤드 y < 손목 y → 헤드가 손목보다 위 → 라켓이 위를 향함 → 폴트
    # (커스텀 YOLO 없으면 감지 불가 — 팔꿈치 근사는 부정확하므로 제거)
    if racket and wrist:
        shaft_diff = wrist[1] - racket[1]   # 양수 = 헤드가 손목보다 위 → 폴트
        result["shaft_fault"] = shaft_diff > margin
        result["details"]["racket_y"]   = round(racket[1], 1)
        result["details"]["wrist_y"]    = round(wrist[1], 1)
        result["details"]["shaft_diff"] = round(shaft_diff, 1)

    return result


# ── 쉐이크 폴트 (9.1.8) ───────────────────────────────────
def detect_shake_fault(wrist_positions, impact_frame, pre_window=15, min_reversals=2,
                       min_reversal_px=15):
    """
    방향 전환 감지 시 최소 이동거리(min_reversal_px) 조건 추가.
    작은 노이즈성 움직임은 방향 전환으로 카운트하지 않음.
    """
    frames = sorted([
        f for f in range(max(0, impact_frame - pre_window), impact_frame + 1)
        if f in wrist_positions and wrist_positions[f]
    ])
    if len(frames) < 4:
        return False

    xs = [wrist_positions[f][0] for f in frames]
    reversals  = 0
    prev_dir   = None
    segment_dx = 0.0   # 현재 방향으로 누적 이동량

    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if abs(dx) < 3:   # 프레임간 노이즈 무시
            continue
        d = 1 if dx > 0 else -1
        if prev_dir is None:
            prev_dir   = d
            segment_dx = dx
        elif d == prev_dir:
            segment_dx += dx
        else:
            # 방향이 바뀜 — 이전 세그먼트가 min_reversal_px 이상일 때만 카운트
            if abs(segment_dx) >= min_reversal_px:
                reversals += 1
            prev_dir   = d
            segment_dx = dx

    return reversals >= min_reversals


# ── 발 폴트 (9.1.4) ───────────────────────────────────────
def detect_foot_fault(frames_data, impact_frame, pre_window=15,
                      calib=None, move_threshold_px=80, service_line_y=None,
                      sustained_frames=5):
    """
    발목 노이즈 제거를 위해:
    1. 서브 윈도우 첫 3프레임 중앙값을 기준점으로 사용 (calib 보다 안정적)
    2. sustained_frames 이상 연속으로 임계값 초과해야 폴트
    """
    frames = sorted([
        f for f in range(max(0, impact_frame - pre_window), impact_frame + 1)
        if f in frames_data
    ])
    if len(frames) < sustained_frames + 3:
        return False, False

    positions = []
    for f in frames:
        kps = frames_data[f]
        pts = [p for p in [get_kp(kps, "left_ankle"), get_kp(kps, "right_ankle")] if p]
        if pts:
            positions.append((
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            ))
    if len(positions) < sustained_frames + 3:
        return False, False

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # 첫 3프레임 중앙값을 기준점으로 (포즈 노이즈 평균화)
    ref_x = float(np.median(xs[:3]))
    ref_y = float(np.median(ys[:3]))

    # sustained_frames 이상 연속으로 임계값 초과해야 폴트
    out_of_range = [
        ((x - ref_x)**2 + (y - ref_y)**2) ** 0.5 > move_threshold_px
        for x, y in zip(xs[3:], ys[3:])
    ]
    consecutive = max_consecutive = 0
    for v in out_of_range:
        consecutive = consecutive + 1 if v else 0
        max_consecutive = max(max_consecutive, consecutive)

    move_fault = max_consecutive >= sustained_frames
    line_fault  = (service_line_y is not None and
                   any(y >= service_line_y for y in ys))

    return move_fault, line_fault


# ── 서브 시작 프레임 감지 ─────────────────────────────────
def find_serve_start(wrist_positions, speeds_global, impact_frame, fps,
                     max_lookback_sec=3.0):
    """
    임팩트 전 손목이 '정지 → 움직임' 전환되는 지점 = 서브 시작.
    speeds_global: find_impact_frames에서 계산된 전체 속도 딕셔너리.
    """
    max_lookback = int(max_lookback_sec * fps)
    search_start = max(0, impact_frame - max_lookback)

    frames = sorted([
        f for f in range(search_start, impact_frame)
        if f in wrist_positions and wrist_positions[f] and f in speeds_global
    ])
    if len(frames) < 4:
        return search_start

    # 해당 구간 속도의 하위 25% = 정지 기준
    seg_speeds = [speeds_global[f] for f in frames]
    low_thresh  = float(np.percentile(seg_speeds, 25))

    # 임팩트에서 뒤로 가며 처음으로 '저속' 구간 진입 = 서브 시작 직전
    for f in reversed(frames):
        if speeds_global[f] <= low_thresh:
            return f

    return search_start


# ── 임팩트 자동 감지 ───────────────────────────────────────
def find_impact_frames(wrist_positions, fps, min_gap_sec=2.0):
    """impacts 리스트와 전체 속도 딕셔너리를 함께 반환."""
    if len(wrist_positions) < 3:
        return [], {}
    frames = sorted(wrist_positions.keys())
    speeds = {}
    for i in range(1, len(frames) - 1):
        f, pf = frames[i], frames[i - 1]
        if wrist_positions[f] and wrist_positions[pf]:
            dx = wrist_positions[f][0] - wrist_positions[pf][0]
            dy = wrist_positions[f][1] - wrist_positions[pf][1]
            speeds[f] = (dx**2 + dy**2) ** 0.5
        else:
            speeds[f] = 0
    if not speeds:
        return [], {}
    vals = list(speeds.values())
    threshold = np.mean(vals) + 1.5 * np.std(vals)
    min_gap   = int(min_gap_sec * fps)
    impacts, last = [], -min_gap
    for f in frames:
        if f in speeds and speeds[f] >= threshold and (f - last) >= min_gap:
            impacts.append(f)
            last = f
    return impacts, speeds


# ── 헛치기 감지 (9.1.10) ──────────────────────────────────
def detect_miss(shuttle_positions, impact_frame, post_window=15, min_displacement=30):
    post_frames = sorted([
        f for f in range(impact_frame, impact_frame + post_window + 1)
        if f in shuttle_positions and shuttle_positions[f] is not None
    ])
    if len(post_frames) < 3:
        return None
    p0, p1 = shuttle_positions[post_frames[0]], shuttle_positions[post_frames[-1]]
    return ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5 < min_displacement


# ── 서브 구간 폴트 분석 ────────────────────────────────────
def analyze_serve(frames_data, impact_frame, serve_start_frame, side="right",
                  det_data=None, shuttle_positions=None, wrist_positions=None,
                  calib=None, service_line_y=None,
                  shake_reversals=2, foot_move_px=80):
    """
    판별 시점 분리:
    - 서브 시작(serve_start_frame): 선밟기, 1.10m 높이
    - 서브 시작 ~ 임팩트 전체: 발 이동, 쉐이크
    - 임팩트(impact_frame): 웨이스트, 샤프트
    """
    # ── 임팩트 시점 판별: 웨이스트, 샤프트 ────────────────
    waist_faults = []; shaft_faults = []
    max_waist_m  = -9999

    # 임팩트 전후 3프레임 윈도우 (임팩트 순간 집중)
    impact_window = range(max(serve_start_frame, impact_frame - 3), impact_frame + 1)
    for f in impact_window:
        if f not in frames_data:
            continue
        shuttle = det_data[f][0] if (det_data and f in det_data) else None
        racket  = det_data[f][1] if (det_data and f in det_data) else None
        fault   = detect_faults(frames_data[f], side, shuttle, racket, calib)
        if fault["waist_fault"]:  waist_faults.append(f)
        if fault["shaft_fault"]:  shaft_faults.append(f)
        if fault["waist_margin"] is not None:
            max_waist_m = max(max_waist_m, fault["waist_margin"])

    # ── 서브 시작 시점 판별: 선밟기, 1.10m 높이 ──────────
    line_fault   = False
    height_fault = False

    if serve_start_frame in frames_data:
        kps_start = frames_data[serve_start_frame]

        # 선밟기: 서브 시작 시 발목이 서비스 라인을 넘었는지
        if service_line_y is not None:
            la = get_kp(kps_start, "left_ankle")
            ra = get_kp(kps_start, "right_ankle")
            for ankle in [la, ra]:
                if ankle and ankle[1] >= service_line_y:
                    line_fault = True

        # 1.10m 높이: 서브 시작 시 셔틀콕 높이 (커스텀 YOLO 있을 때만)
        if det_data and serve_start_frame in det_data:
            shuttle_start = det_data[serve_start_frame][0]
            h_thresh = (calib or {}).get("height_thresh_y")
            if shuttle_start and h_thresh:
                body_h = (calib or {}).get("body_height") or 200.0
                margin = body_h * 0.02
                if (h_thresh - shuttle_start[1]) > margin:
                    height_fault = True

    # ── 서브 전체 구간 판별: 발 이동, 쉐이크 ─────────────
    pre_window = impact_frame - serve_start_frame
    move_fault, _ = detect_foot_fault(
        frames_data, impact_frame, pre_window, calib, foot_move_px,
        service_line_y=None   # 선밟기는 위에서 별도 처리
    )
    shake = (detect_shake_fault(wrist_positions, impact_frame, pre_window, shake_reversals)
             if wrist_positions else False)

    # ── 헛치기 ────────────────────────────────────────────
    miss = detect_miss(shuttle_positions, impact_frame) if shuttle_positions else None

    n = max(len(list(impact_window)), 1)
    return {
        "impact_frame":         impact_frame,
        "serve_start_frame":    serve_start_frame,
        "impact_time_sec":      None,
        "serve_start_time_sec": None,
        "waist_fault":          len(waist_faults) / n >= 0.5,
        "height_fault":         height_fault,
        "shaft_fault":          len(shaft_faults) >= 2,
        "shake_fault":          shake,
        "foot_move_fault":      move_fault,
        "foot_line_fault":      line_fault,
        "miss_fault":           miss,
        "waist_fault_frames":   len(waist_faults),
        "shaft_fault_frames":   len(shaft_faults),
        "max_waist_margin_px":  round(max_waist_m, 1) if max_waist_m != -9999 else None,
    }


# ── 기준선 드로잉 (항상 표시) ─────────────────────────────
def draw_reference_lines(frame, calib, service_line_y=None):
    h, w = frame.shape[:2]

    if calib.get("waist_y"):
        wy = int(calib["waist_y"])
        cv2.line(frame, (0, wy), (w, wy), (0, 220, 220), 2)
        cv2.putText(frame, "WAIST (9.1.6)", (8, wy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

    if calib.get("height_thresh_y"):
        ht = int(calib["height_thresh_y"])
        cv2.line(frame, (0, ht), (w, ht), (0, 140, 255), 2)
        cv2.putText(frame, "1.10m (9.1.6.2)", (8, ht - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    if service_line_y is not None:
        sl = int(service_line_y)
        cv2.line(frame, (0, sl), (w, sl), (0, 0, 220), 2)
        cv2.putText(frame, "SERVICE LINE (9.1.4)", (8, sl - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)


# ── 서브 결과 표시 (임팩트 후 N초) ───────────────────────
def draw_serve_result(frame, serve_result, shuttle=None, racket=None):
    """임팩트 후 serve_result를 화면 중앙에 크게 표시."""
    h, w = frame.shape[:2]

    tags = []
    if serve_result.get("waist_fault"):     tags.append("WAIST")
    if serve_result.get("height_fault"):    tags.append("HEIGHT 1.10m")
    if serve_result.get("shaft_fault"):     tags.append("SHAFT")
    if serve_result.get("shake_fault"):     tags.append("SHAKE")
    if serve_result.get("foot_move_fault"): tags.append("FOOT MOVE")
    if serve_result.get("foot_line_fault"): tags.append("FOOT LINE")
    if serve_result.get("miss_fault"):      tags.append("MISS")

    is_fault = bool(tags)
    bg_color   = (0, 0, 180)   if is_fault else (0, 140, 0)
    label_main = ("FAULT"      if is_fault else "OK")
    label_sub  = " | ".join(tags) if tags else "LEGAL SERVE"

    # 반투명 배경 박스
    box_y1, box_y2 = h // 2 - 70, h // 2 + 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, box_y1), (w, box_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # 메인 텍스트
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(label_main, font, 2.0, 4)
    cv2.putText(frame, label_main, ((w - tw) // 2, h // 2 - 5),
                font, 2.0, (255, 255, 255), 4)

    # 서브 텍스트 (폴트 종류)
    (tw2, _), _ = cv2.getTextSize(label_sub, font, 0.75, 2)
    cv2.putText(frame, label_sub, ((w - tw2) // 2, h // 2 + 50),
                font, 0.75, (220, 220, 220), 2)

    # 셔틀콕 / 라켓헤드 검출 표시
    if shuttle:
        sx, sy = int(shuttle[0]), int(shuttle[1])
        cv2.circle(frame, (sx, sy), 14, (0, 255, 0), 2)
    if racket:
        rx1, ry1, rx2, ry2 = int(racket[3]), int(racket[4]), int(racket[5]), int(racket[6])
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)


# ── 타임스탬프 ─────────────────────────────────────────────
def draw_timestamp(frame, frame_idx, fps):
    h, w = frame.shape[:2]
    ts = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"{ts:.2f}s  f={frame_idx}",
                (w - 205, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ── 메인 ──────────────────────────────────────────────────
def analyze_video(input_path, output_path=None, side="right",
                  pre_window=15, expected_serves=None,
                  det_model_path=None, player_height_m=1.70,
                  calibration_frames=90,
                  service_line_y=None,
                  shake_reversals=2,
                  foot_move_px=20,
                  result_display_sec=3.0):

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"파일 없음: {input_path}")
    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_result{input_path.suffix}")

    pose_model = YOLO("yolov8n-pose.pt")
    det_model  = YOLO(det_model_path) if det_model_path else None
    print(f"커스텀 YOLO: {det_model_path}" if det_model else
          "커스텀 YOLO 없음 → 웨이스트/쉐이크/발 폴트만 감지 (샤프트·높이·헛치기는 YOLO 필요)")

    cap   = cv2.VideoCapture(str(input_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[1/3] 포즈 + 객체 추출 중... ({total}프레임 / {fps:.0f}fps)")

    frames_data = {}; det_data = {}; wrist_positions = {}; shuttle_positions = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose_res = pose_model(frame, verbose=False)
        if pose_res[0].keypoints is not None and len(pose_res[0].keypoints) > 0:
            kps = pose_res[0].keypoints.data[0].cpu().numpy()
            frames_data[frame_idx]     = kps
            wrist_positions[frame_idx] = get_kp(kps, f"{side}_wrist")
        if det_model:
            sh, rk = detect_objects(det_model, frame)
            det_data[frame_idx] = (sh, rk)
            shuttle_positions[frame_idx] = (sh[0], sh[1]) if sh else None
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)")
    cap.release()

    print(f"\n[보정] 처음 {calibration_frames}프레임으로 허리선·1.10m선 고정 중...")
    calib = calibrate_body(frames_data, calibration_frames, player_height_m)
    print(f"  허리선:  {calib['waist_y']:.0f}px" if calib['waist_y'] else "  허리선: 감지 실패")
    print(f"  1.10m선: {calib['height_thresh_y']:.0f}px" if calib['height_thresh_y'] else "  1.10m선: 감지 실패")
    if service_line_y:
        print(f"  서비스라인: {service_line_y}px")

    print(f"\n[2/3] 임팩트 + 서브 시작 감지 중...")
    impacts, speeds_global = find_impact_frames(wrist_positions, fps, min_gap_sec=2.0)
    print(f"  감지: {len(impacts)}개" + (f" / 예상: {expected_serves}개" if expected_serves else ""))
    if expected_serves and abs(len(impacts) - expected_serves) > 3:
        print("  ⚠️  예상 수와 차이납니다. min_gap_sec 조정을 시도해보세요.")

    serves = []
    for imp in impacts:
        serve_start = find_serve_start(wrist_positions, speeds_global, imp, fps)
        res = analyze_serve(
            frames_data, imp, serve_start, side,
            det_data=det_data if det_model else None,
            shuttle_positions=shuttle_positions if det_model else None,
            wrist_positions=wrist_positions,
            calib=calib, service_line_y=service_line_y,
            shake_reversals=shake_reversals, foot_move_px=foot_move_px,
        )
        res["impact_time_sec"]      = round(imp / fps, 2)
        res["serve_start_time_sec"] = round(serve_start / fps, 2)
        serves.append(res)

    # ── Pass 3: 결과 영상 생성 (YOLO 재실행 없음) ─────────
    print(f"\n[3/3] 결과 영상 생성 중...")
    cap    = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    impact_set = set(impacts)
    result_display_frames = int(result_display_sec * fps)

    # 임팩트 프레임 → 서브 결과 매핑
    impact_to_result = {s["impact_frame"]: s for s in serves}
    active_result = None   # (serve_dict, expire_frame)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 임팩트 프레임 도달 시 결과 활성화
        if frame_idx in impact_set:
            active_result = (impact_to_result[frame_idx],
                             frame_idx + result_display_frames)

        # 결과 표시 만료 체크
        if active_result and frame_idx > active_result[1]:
            active_result = None

        if frame_idx in frames_data:
            kps     = frames_data[frame_idx]
            shuttle = det_data[frame_idx][0] if (det_model and frame_idx in det_data) else None
            racket  = det_data[frame_idx][1] if (det_model and frame_idx in det_data) else None

            draw_skeleton(frame, kps)
            draw_reference_lines(frame, calib, service_line_y)

            if active_result:
                draw_serve_result(frame, active_result[0], shuttle, racket)

        draw_timestamp(frame, frame_idx, fps)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # 리포트
    fault_serves = [s for s in serves if
                    s["waist_fault"] or s["height_fault"] or s["shaft_fault"] or
                    s["shake_fault"] or s["foot_move_fault"] or s["foot_line_fault"] or
                    s.get("miss_fault")]
    report = {
        "input": str(input_path), "fps": fps, "total_frames": total,
        "side": side, "player_height_m": player_height_m,
        "det_model": det_model_path,
        "calibration": {
            "frames_used": calibration_frames,
            "waist_y_px": calib["waist_y"],
            "height_thresh_px": calib["height_thresh_y"],
            "service_line_y": service_line_y,
        },
        "pre_impact_window_frames": pre_window,
        "total_serves": len(serves),
        "fault_serves": len(fault_serves),
        "ok_serves":    len(serves) - len(fault_serves),
        "serves": serves,
    }
    report_path = str(Path(output_path).with_suffix(".json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*45}")
    print(f"완료!  결과: {output_path}")
    print(f"총 {len(serves)}개  /  폴트 {len(fault_serves)}개  /  정상 {len(serves)-len(fault_serves)}개\n")
    for i, s in enumerate(serves):
        tags = []
        if s["waist_fault"]:       tags.append("웨이스트")
        if s["height_fault"]:      tags.append("높이1.10m")
        if s["shaft_fault"]:       tags.append("샤프트")
        if s["shake_fault"]:       tags.append("쉐이크")
        if s["foot_move_fault"]:   tags.append("발이동")
        if s["foot_line_fault"]:   tags.append("라인밟기")
        if s.get("miss_fault"):    tags.append("헛치기")
        print(f"  서브 {i+1:02d}  {s['impact_time_sec']:6.2f}s  →  {' / '.join(tags) or '정상'}")
    return report


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="배드민턴 서브 폴트 검출 v5")
    parser.add_argument("input")
    parser.add_argument("-o", "--output",            default=None)
    parser.add_argument("-s", "--side",              default="right", choices=["right","left"])
    parser.add_argument("-w", "--pre_window",        type=int,   default=15)
    parser.add_argument("-n", "--serves",            type=int,   default=None)
    parser.add_argument("--det_model",               default=None)
    parser.add_argument("--player_height",           type=float, default=1.70)
    parser.add_argument("--calibration_frames",      type=int,   default=90)
    parser.add_argument("--service_line_y",          type=float, default=None)
    parser.add_argument("--shake_reversals",         type=int,   default=2)
    parser.add_argument("--foot_move_px",            type=int,   default=80)
    parser.add_argument("--result_display_sec",      type=float, default=3.0,
                        help="서브 결과 표시 지속 시간(초) (default: 3.0)")
    args = parser.parse_args()

    analyze_video(
        args.input, args.output, args.side,
        args.pre_window, args.serves,
        args.det_model, args.player_height,
        args.calibration_frames, args.service_line_y,
        args.shake_reversals, args.foot_move_px,
        args.result_display_sec,
    )
