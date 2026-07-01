"""
배드민턴 서브 폴트 라이브 검출기
"""

import cv2
import argparse
import numpy as np
import subprocess
import threading
from collections import deque
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import platform
import torch

# ── 한글 폰트 ────────────────────────────────────────────────
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="0")
    parser.add_argument("--det_model",  default="best_v9.pt")
    parser.add_argument("--skip",       type=int, default=1)
    parser.add_argument("--infer_size", type=int, default=320)
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    det_model = YOLO(args.det_model) if Path(args.det_model).exists() else None
    if not det_model:
        print("경고: 모델 없음"); return
    print(f"모델: {args.det_model}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det_model.to(device)
    print(f"디바이스: {device}")

    backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    shuttle         = None
    frame_idx       = 0
    calib_locked    = False
    last_beep_frame = -999
    shuttle_history = deque(maxlen=10)
    MOVE_THRESH     = 60

    def beep():
        if platform.system() == "Darwin":
            subprocess.Popen(["say", "-v", "Flo", "fault"])
        else:
            print("\a", end="", flush=True)

    ret0, f0 = cap.read()
    calib_y_val  = [f0.shape[0] // 2 if ret0 else 360]
    thresh_y_val = [None]

    win_name = "서브폴트 — 마우스클릭:선이동  Enter:확정  r:재설정  q:종료"
    cv2.namedWindow(win_name)

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON):
            calib_y_val[0] = y
            if calib_locked:
                thresh_y_val[0] = float(y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            calib_y_val[0] = y
            if calib_locked:
                thresh_y_val[0] = float(y)

    cv2.setMouseCallback(win_name, on_mouse)
    print("마우스 클릭으로 선 위치 조절 → Enter 로 1.15m 확정  |  r 재설정  |  q 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        h, w = frame.shape[:2]

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

        line_y = int(thresh_y_val[0] if calib_locked else calib_y_val[0])
        if calib_locked:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 120, 255), 4)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 200, 255), 2)
            cv2.putText(frame, "1.15m", (8, line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3)
        else:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 255), 4)
            put_ko(frame, "클릭/드래그로 선 이동  →  Enter 로 확정",
                   (10, line_y - 35), size=28, color=(0, 220, 255))

        if shuttle:
            shuttle_history.append((shuttle[0], shuttle[1]))
        total_move = 0.0
        if len(shuttle_history) >= 4:
            xs = [p[0] for p in shuttle_history]
            ys = [p[1] for p in shuttle_history]
            total_move = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
        is_moving = total_move > MOVE_THRESH

        if shuttle:
            cx, cy = int(shuttle[0]), int(shuttle[1])
            if not calib_locked or is_moving:
                color = (180, 180, 180)
            elif cy < thresh_y_val[0]:
                color = (0, 0, 255)
                if frame_idx - last_beep_frame > 60:
                    threading.Thread(target=beep, daemon=True).start()
                    last_beep_frame = frame_idx
            else:
                color = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 18, color, 3)
            if calib_locked:
                cv2.putText(frame, f"{'moving' if is_moving else 'hold'} {total_move:.0f}px",
                            (cx + 22, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):
            fs = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fs != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
        elif key == 13:
            thresh_y_val[0] = float(calib_y_val[0])
            calib_locked = True
            print(f"1.15m 기준 확정: y={thresh_y_val[0]:.0f}")
        elif key == ord('r'):
            calib_locked = False
            thresh_y_val[0] = None
            print("기준선 재설정")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
