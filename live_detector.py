"""
배드민턴 서브 폴트 라이브 검출기
사용법:
  python live_detector.py                        # 웹캠 (카메라 0)
  python live_detector.py --source 1             # 두 번째 카메라
  python live_detector.py --source video.mp4     # 영상 파일 테스트
  python live_detector.py --det_model best_v9.pt
"""

import cv2
import argparse
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# ── 한글 폰트 ────────────────────────────────────────────────
import platform
if platform.system() == "Darwin":
    _KO_FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:
    _KO_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
_ko_fonts = {}

def _ko_font(size):
    if size not in _ko_fonts:
        try:   _ko_fonts[size] = ImageFont.truetype(_KO_FONT_PATH, size)
        except: _ko_fonts[size] = ImageFont.load_default()
    return _ko_fonts[size]

def put_ko(frame, text, pos, size=28, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=_ko_font(size), fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── COCO keypoint 인덱스 ────────────────────────────────────
KP = {
    "nose": 0,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}
CONF_THR    = 0.4
SHUTTLE_CLS = 0
RACKET_CLS  = 1

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,5),(0,6),
]


def get_kp(kps, name):
    idx = KP[name]
    x, y, c = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    return (x, y) if c >= CONF_THR else None


def body_metrics(kps):
    ls = get_kp(kps, "left_shoulder");  rs = get_kp(kps, "right_shoulder")
    lh = get_kp(kps, "left_hip");       rh = get_kp(kps, "right_hip")
    la = get_kp(kps, "left_ankle");     ra = get_kp(kps, "right_ankle")

    def avg_y(a, b):
        if a and b: return (a[1]+b[1])/2
        return (a or b or [None,None])[1]

    shoulder_y = avg_y(ls, rs)
    hip_y      = avg_y(lh, rh)
    ankle_y    = avg_y(la, ra)
    waist_y    = (shoulder_y + (hip_y - shoulder_y)*0.60) if (shoulder_y and hip_y) else None
    body_h     = (ankle_y - shoulder_y) if (shoulder_y and ankle_y) else None

    return {
        "shoulder_y": shoulder_y, "hip_y": hip_y,
        "waist_y": waist_y, "ankle_y": ankle_y,
        "body_height": body_h,
        "valid": all([shoulder_y, hip_y, waist_y, ankle_y,
                      body_h and body_h > 50]),
    }


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
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if cls == SHUTTLE_CLS and (shuttle is None or conf > shuttle[2]):
            shuttle = (cx, cy, conf, x1, y1, x2, y2)
        elif cls == RACKET_CLS and (racket is None or conf > racket[2]):
            racket  = (cx, cy, conf, x1, y1, x2, y2)
    return shuttle, racket


# ── 롤링 버퍼 기반 임팩트 감지 ──────────────────────────────
class ServeDetector:
    def __init__(self, fps, side="right",
                 min_gap_sec=2.0, impact_speed_factor=1.5,
                 window_sec=1.5, shake_reversals=2):
        self.fps             = fps
        self.side            = side
        self.min_gap_frames  = int(min_gap_sec * fps)
        self.speed_factor    = impact_speed_factor
        self.window          = int(window_sec * fps)
        self.shake_reversals = shake_reversals

        # 롤링 버퍼
        self.wrist_buf   = deque(maxlen=self.window * 2)  # (frame_idx, pos)
        self.shuttle_buf = deque(maxlen=self.window * 2)
        self.racket_buf  = deque(maxlen=self.window * 2)
        self.kps_buf     = deque(maxlen=self.window * 2)
        self.speed_buf   = deque(maxlen=self.window * 2)

        self.last_impact   = -self.min_gap_frames
        self.speed_history = deque(maxlen=120)  # 속도 통계용

        # 보정값 (처음 90프레임)
        self.calib          = None
        self.calib_frames   = []
        self.calib_done     = False
        self.calib_target   = 90

    def update_calib(self, kps, frame_idx):
        if self.calib_done or frame_idx > self.calib_target:
            if not self.calib_done and self.calib_frames:
                self._finalize_calib()
            return
        m = body_metrics(kps)
        if m["valid"]:
            self.calib_frames.append(m)
        if frame_idx == self.calib_target:
            self._finalize_calib()

    def _finalize_calib(self):
        if not self.calib_frames:
            self.calib_done = True
            return
        waist_ys  = [m["waist_y"]     for m in self.calib_frames]
        body_hs   = [m["body_height"] for m in self.calib_frames]
        ankle_ys  = [m["ankle_y"]     for m in self.calib_frames]
        self.calib = {
            "waist_y":     float(np.median(waist_ys)),
            "body_height": float(np.median(body_hs)),
            "ankle_y_ref": float(np.median(ankle_ys)),
            "height_thresh_y": None,  # 신장 없으면 None
        }
        self.calib_done = True

    def push(self, frame_idx, kps, shuttle, racket):
        self.update_calib(kps, frame_idx)

        wrist = get_kp(kps, f"{self.side}_wrist") if kps is not None else None
        self.wrist_buf.append((frame_idx, wrist))
        self.shuttle_buf.append((frame_idx, shuttle))
        self.racket_buf.append((frame_idx, racket))
        self.kps_buf.append((frame_idx, kps))

        # 손목 속도 계산
        speed = 0.0
        if len(self.wrist_buf) >= 2:
            fi, pi = self.wrist_buf[-1]
            fj, pj = self.wrist_buf[-2]
            if pi and pj:
                speed = ((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2)**0.5
        self.speed_buf.append((frame_idx, speed))
        self.speed_history.append(speed)

    def check_impact(self, frame_idx):
        if frame_idx - self.last_impact < self.min_gap_frames:
            return None
        if len(self.speed_history) < 20:
            return None

        vals = list(self.speed_history)
        threshold = np.mean(vals) + self.speed_factor * np.std(vals)
        fi, speed = self.speed_buf[-1]
        if speed < threshold:
            return None

        # 임팩트 확정
        self.last_impact = frame_idx
        return self._analyze(frame_idx)

    def _analyze(self, impact_frame):
        # 서브 시작 = 롤링 버퍼 내 저속 구간 시작점
        speeds = [(f, s) for f, s in self.speed_buf]
        if len(speeds) < 4:
            return None

        seg_speeds = [s for _, s in speeds]
        low_thresh = float(np.percentile(seg_speeds, 25))
        serve_start = speeds[0][0]
        for f, s in reversed(speeds[:-5]):
            if s <= low_thresh:
                serve_start = f
                break

        # 각 버퍼를 dict로 변환
        wrist_dict   = {f: p for f, p in self.wrist_buf}
        shuttle_dict = {f: s for f, s in self.shuttle_buf}
        racket_dict  = {f: r for f, r in self.racket_buf}
        kps_dict     = {f: k for f, k in self.kps_buf}

        calib   = self.calib or {}
        body_h  = calib.get("body_height") or 200.0
        waist_y = calib.get("waist_y")
        margin  = body_h * 0.02

        # ── 웨이스트 폴트 (임팩트 ±3프레임, 셔틀 상단) ──────
        waist_fault = False
        for f in range(max(serve_start, impact_frame-3), impact_frame+1):
            sh = shuttle_dict.get(f)
            w  = wrist_dict.get(f)
            impact_y = sh[4] if sh else (w[1] if w else None)
            if impact_y and waist_y:
                if (waist_y - impact_y) > margin:
                    waist_fault = True

        # ── 높이 폴트 (서브 시작 시 셔틀 높이) ──────────────
        height_fault = False
        h_thresh = calib.get("height_thresh_y")
        if h_thresh:
            sh_start = shuttle_dict.get(serve_start)
            if sh_start and (h_thresh - sh_start[1]) > margin:
                height_fault = True

        # ── 샤프트 폴트 (서브시작~임팩트, 라켓헤드 < 손목) ──
        shaft_fault = False
        shaft_count = 0
        for f in range(serve_start, impact_frame+1):
            rk = racket_dict.get(f)
            w  = wrist_dict.get(f)
            if rk and w:
                if (w[1] - rk[1]) > margin:
                    shaft_count += 1
        if shaft_count >= 2:
            shaft_fault = True

        # ── 쉐이크 폴트 (서브 구간 손목 x 방향 전환) ────────
        shake_fault = False
        xs = [wrist_dict[f][0] for f in range(serve_start, impact_frame+1)
              if f in wrist_dict and wrist_dict[f]]
        if len(xs) >= 4:
            reversals = 0; prev_dir = None; seg_dx = 0.0
            for i in range(1, len(xs)):
                dx = xs[i] - xs[i-1]
                if abs(dx) < 3: continue
                d = 1 if dx > 0 else -1
                if prev_dir is None:
                    prev_dir = d; seg_dx = dx
                elif d == prev_dir:
                    seg_dx += dx
                else:
                    if abs(seg_dx) >= 15: reversals += 1
                    prev_dir = d; seg_dx = dx
            shake_fault = reversals >= self.shake_reversals

        # ── 헛치기 (임팩트 후 셔틀 이동 없음) ───────────────
        miss_fault = None
        post = [(f, shuttle_dict[f]) for f in range(impact_frame, impact_frame+16)
                if f in shuttle_dict and shuttle_dict[f]]
        if len(post) >= 3:
            p0, p1 = post[0][1], post[-1][1]
            dist = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5
            miss_fault = dist < 30

        return {
            "impact_frame":    impact_frame,
            "serve_start":     serve_start,
            "waist_fault":     waist_fault,
            "height_fault":    height_fault,
            "shaft_fault":     shaft_fault,
            "shake_fault":     shake_fault,
            "miss_fault":      miss_fault,
            "shuttle":         shuttle_dict.get(impact_frame),
            "racket":          racket_dict.get(impact_frame),
        }


# ── 결과 화면 출력 ───────────────────────────────────────────
def draw_result(frame, result):
    h, w = frame.shape[:2]
    tags = []
    if result.get("waist_fault"):  tags.append("웨이스트")
    if result.get("height_fault"): tags.append("높이1.15m")
    if result.get("shaft_fault"):  tags.append("샤프트")
    if result.get("shake_fault"):  tags.append("쉐이크")
    if result.get("miss_fault"):   tags.append("헛치기")

    is_fault = bool(tags)
    if not is_fault:
        return  # 정상은 표시 안 함

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2-80), (w, h//2+80), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    put_ko(frame, "폴트", (w//2 - 60, h//2 - 55), size=72,
           color=(255,255,255))
    put_ko(frame, " / ".join(tags), (20, h//2 + 30), size=32,
           color=(220,220,220))

    sh = result.get("shuttle")
    rk = result.get("racket")
    if sh:
        cv2.circle(frame, (int(sh[0]), int(sh[1])), 14, (0,255,0), 2)
    if rk:
        cv2.rectangle(frame, (int(rk[3]),int(rk[4])),
                      (int(rk[5]),int(rk[6])), (0,165,255), 2)


def draw_hud(frame, calib, shuttle, racket):
    h, w = frame.shape[:2]
    if calib and calib.get("waist_y"):
        wy = int(calib["waist_y"])
        cv2.line(frame, (0,wy), (w,wy), (0,220,220), 2)
        cv2.putText(frame, "WAIST (↑↓조절)", (8, wy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,220), 1)
    if calib and calib.get("height_thresh_y"):
        ht = int(calib["height_thresh_y"])
        cv2.line(frame, (0,ht), (w,ht), (0,140,255), 2)
        cv2.putText(frame, "1.15m", (8, ht-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,140,255), 1)
    if shuttle:
        cv2.circle(frame, (int(shuttle[0]),int(shuttle[1])), 12, (0,255,0), 2)
    if racket:
        cv2.rectangle(frame, (int(racket[3]),int(racket[4])),
                      (int(racket[5]),int(racket[6])), (0,165,255), 2)


# ── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="배드민턴 서브 폴트 라이브 검출기")
    parser.add_argument("--source",     default="0")
    parser.add_argument("--det_model",  default="best_v9.pt")
    parser.add_argument("--skip",       type=int, default=2)
    parser.add_argument("--infer_size", type=int, default=640)
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    det_model = YOLO(args.det_model) if Path(args.det_model).exists() else None
    if not det_model:
        print("경고: 모델 없음"); return
    print(f"모델: {args.det_model}")

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    shuttle      = None
    frame_idx    = 0
    calib_locked = False

    ret0, f0 = cap.read()
    calib_y = f0.shape[0] // 2 if ret0 else 360
    thresh_y = None

    win_name = "서브폴트 — 마우스클릭:선이동  Enter:확정  r:재설정  q:종료"
    cv2.namedWindow(win_name)

    def on_mouse(event, x, y, flags, param):
        nonlocal calib_y, thresh_y
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            calib_y = y
            if calib_locked:
                thresh_y = float(y)

    cv2.setMouseCallback(win_name, on_mouse)
    print("마우스 클릭으로 선 위치 조절 → Enter 로 1.15m 확정  |  r 재설정  |  q 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        h, w = frame.shape[:2]

        # ── 셔틀콕 감지 ──────────────────────────────────────
        if frame_idx % args.skip == 0:
            scale = args.infer_size / max(h, w)
            small = cv2.resize(frame, (int(w*scale), int(h*scale)))
            res   = det_model(small, verbose=False)
            shuttle = None
            if res[0].boxes is not None:
                for box in res[0].boxes:
                    if float(box.conf[0]) < 0.25:
                        continue
                    x1,y1,x2,y2 = [float(v)/scale for v in box.xyxy[0]]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    conf = float(box.conf[0])
                    if shuttle is None or conf > shuttle[2]:
                        shuttle = (cx, cy, conf)

        # ── 기준선 그리기 ─────────────────────────────────────
        line_y = int(thresh_y if calib_locked else calib_y)
        if calib_locked:
            # 확정된 기준선 — 주황색
            cv2.line(frame, (0, line_y), (w, line_y), (0, 140, 255), 2)
            cv2.putText(frame, "1.15m", (8, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        else:
            # 조절 중 기준선 — 노란색 점선 느낌
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 255), 2)
            put_ko(frame, "↑↓ 로 선 이동  →  Enter 로 1.15m 확정",
                   (10, line_y - 30), size=26, color=(0, 220, 255))

        # ── 셔틀콕 원 그리기 ──────────────────────────────────
        if shuttle:
            cx, cy = int(shuttle[0]), int(shuttle[1])
            if not calib_locked:
                color = (200, 200, 200)  # 캘리브레이션 전 — 회색
            elif cy < thresh_y:
                color = (0, 0, 255)      # 빨강 — 1.15m 초과 (폴트)
            else:
                color = (0, 255, 0)      # 초록 — 정상
            cv2.circle(frame, (cx, cy), 18, color, 3)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):  # 전체화면 토글
            fs = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        elif key == 13:   # Enter
            thresh_y    = float(calib_y)
            calib_locked = True
            print(f"1.15m 기준 확정: y={thresh_y:.0f}")
        elif key == ord('r'):
            calib_locked = False
            thresh_y     = None
            print("기준선 재설정")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
