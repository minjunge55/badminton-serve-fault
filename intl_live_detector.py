"""
국제시합 서브 높이 폴트 라이브 검출기 (스레드 분리 버전)
- 카메라 표시: 메인 스레드 (끊김 없음)
- YOLO 추론: 백그라운드 스레드 (비동기)
- 깜빡임 없이 부드러운 화면

사용법:
  python intl_live_detector.py --source 1 --side right
  python intl_live_detector.py --source 1 --side left
  python intl_live_detector.py --source 1 --save
"""

import cv2
import json
import threading
import argparse
import subprocess
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# ── 한글 폰트 ─────────────────────────────────────────────
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

# ── COCO 키포인트 ──────────────────────────────────────────
KP_IDX = {
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}
SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
CONF_KP = 0.4
SHUTTLE_CLASS = 0


def get_kp(kps, name):
    idx = KP_IDX.get(name)
    if idx is None or idx >= len(kps):
        return None
    x, y, conf = kps[idx]
    return (float(x), float(y)) if conf > CONF_KP else None


def calc_height_thresh(kps, player_height_m=1.70):
    la = get_kp(kps, "left_ankle");  ra = get_kp(kps, "right_ankle")
    ls = get_kp(kps, "left_shoulder"); rs = get_kp(kps, "right_shoulder")
    ankle_y    = np.mean([p[1] for p in [la, ra] if p]) if (la or ra) else None
    shoulder_y = np.mean([p[1] for p in [ls, rs] if p]) if (ls or rs) else None
    if ankle_y is None or shoulder_y is None:
        return None
    body_px = ankle_y - shoulder_y
    if body_px < 50:
        return None
    return ankle_y - (1.15 * body_px / (player_height_m * 0.90))


def draw_skeleton(frame, kps):
    if kps is None:
        return
    for i, j in SKELETON:
        if i >= len(kps) or j >= len(kps):
            continue
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci < CONF_KP or cj < CONF_KP:
            continue
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 230, 255), 2)
    for x, y, conf in kps:
        if conf < CONF_KP:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)


def draw_result_box(frame, fault_state, fault_cnt, ok_cnt):
    H, W = frame.shape[:2]
    bw, bh = 220, 115
    x0, y0 = W - bw - 16, H - bh - 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+bw, y0+bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0+bw, y0+bh), (80, 80, 80), 1)
    put_ko(frame, "1.15m 판정",         (x0+10, y0+4),  size=18, color=(200, 200, 200))
    put_ko(frame, f"폴트  {fault_cnt}", (x0+10, y0+28), size=22, color=(80, 80, 255))
    put_ko(frame, f"정상  {ok_cnt}",    (x0+10, y0+54), size=22, color=(80, 255, 80))
    if fault_state is True:
        put_ko(frame, "▶ FAULT", (x0+10, y0+82), size=22, color=(0, 0, 255))
    elif fault_state is False:
        put_ko(frame, "▶ OK",    (x0+10, y0+82), size=22, color=(0, 200, 60))


# ── 공유 상태 (스레드 간 통신) ────────────────────────────
class SharedState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.latest_frame = None   # 추론용 최신 프레임
        self.kps          = None
        self.shuttle      = None
        self.height_thresh= None
        self.calib_done   = False
        self.height_ys    = []
        self.shuttle_hist = deque(maxlen=12)
        self.fault_cnt    = 0
        self.ok_cnt       = 0
        self.current_fault= None
        self.result_expire= 0
        self.frame_count  = 0
        self.running      = True


def inference_worker(pose_model, det_model, state,
                     calib_frames, conf_thr, infer_size,
                     player_height_m, result_frames):
    """백그라운드 스레드: YOLO 추론 전담"""
    while state.running:
        with state.lock:
            frame = state.latest_frame
            if frame is None:
                continue

        # ── 포즈 추정 ──────────────────────────────
        pose_res = pose_model(frame, imgsz=infer_size, verbose=False)
        kps = None
        if pose_res[0].keypoints is not None and len(pose_res[0].keypoints) > 0:
            kps = pose_res[0].keypoints.data[0].cpu().numpy()

        # ── 셔틀콕 감지 ────────────────────────────
        sh = None
        det_res = det_model(frame, imgsz=infer_size, verbose=False, conf=conf_thr)
        if det_res[0].boxes:
            for box in det_res[0].boxes:
                if int(box.cls[0]) == SHUTTLE_CLASS:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    if sh is None or conf > sh[2]:
                        sh = (cx, cy, conf, x1, y1, x2, y2)

        with state.lock:
            state.kps     = kps
            state.shuttle = sh
            state.frame_count += 1
            fc = state.frame_count

            # 캘리브레이션
            if not state.calib_done and kps is not None:
                t = calc_height_thresh(kps, player_height_m)
                if t is not None:
                    state.height_ys.append(t)
                if len(state.height_ys) >= calib_frames:
                    state.height_thresh = float(np.median(state.height_ys))
                    state.calib_done = True
                    print(f"[캘리브레이션 완료] 1.15m 기준선: {state.height_thresh:.0f}px")

            # 정지 판단
            state.shuttle_hist.append((sh[0], sh[1]) if sh else None)
            recent = [p for p in state.shuttle_hist if p is not None]
            stationary = False
            if len(recent) >= 5:
                xs = [p[0] for p in recent]
                ys = [p[1] for p in recent]
                stationary = (max(xs)-min(xs)) < 20 and (max(ys)-min(ys)) < 20

            # 폴트 판정
            if stationary and sh is not None and state.height_thresh is not None:
                shuttle_top_y = sh[4]
                body_h = 200.0
                if kps is not None:
                    la = get_kp(kps, "left_ankle");  ra = get_kp(kps, "right_ankle")
                    ls = get_kp(kps, "left_shoulder"); rs = get_kp(kps, "right_shoulder")
                    ay = np.mean([p[1] for p in [la, ra] if p]) if (la or ra) else None
                    sy = np.mean([p[1] for p in [ls, rs] if p]) if (ls or rs) else None
                    if ay and sy:
                        body_h = ay - sy
                margin   = body_h * 0.02
                is_fault = bool((state.height_thresh - shuttle_top_y) > margin)

                if fc > state.result_expire:
                    state.result_expire  = fc + result_frames
                    state.current_fault  = is_fault
                    if is_fault:
                        state.fault_cnt += 1
                        print(f"[FAULT] 셔틀콕 top={shuttle_top_y:.0f}px / 기준={state.height_thresh:.0f}px")
                    else:
                        state.ok_cnt += 1
                        print(f"[OK]")

            # 판정 만료
            if state.current_fault is not None and fc > state.result_expire:
                state.current_fault = None


def save_calib(path, height_thresh, W, H, fps):
    data = {"height_thresh_px": height_thresh, "W": W, "H": H, "fps": fps}
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[캘리브레이션 저장] {path}  (1.15m = {height_thresh:.0f}px)")


def load_calib(path):
    data = json.loads(Path(path).read_text())
    print(f"[캘리브레이션 불러오기] {path}  (1.15m = {data['height_thresh_px']:.0f}px)")
    return data


def manual_calib(cap, save_path=None):
    """
    화면에서 클릭으로 1.15m 기준선 직접 지정.
    1.15m 막대기 꼭대기를 클릭하면 그 y좌표가 기준선이 됨.
    """
    clicked_y = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_y[0] = y
            print(f"[클릭] y={y}px → 1.15m 기준선 설정")

    WIN = "캘리브레이션 — 1.15m 막대기 꼭대기 클릭 후 Enter"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    print("\n▶ 1.15m 막대기를 서브 위치에 세우고 꼭대기를 클릭하세요.")
    print("  클릭 후 Enter → 확정 / r → 다시 클릭 / q → 취소\n")

    thresh = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # 클릭한 위치에 기준선 미리보기
        if clicked_y[0] is not None:
            cv2.line(frame, (0, clicked_y[0]), (W, clicked_y[0]), (0, 200, 255), 2)
            put_ko(frame, "1.15m (클릭 위치)", (10, clicked_y[0] - 30),
                   size=22, color=(0, 200, 255))

        put_ko(frame, "꼭대기 클릭 → Enter 확정", (10, 30), size=22, color=(255, 200, 0))
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('\r'):  # Enter
            if clicked_y[0] is not None:
                thresh = clicked_y[0]
                break
        elif key == ord('r'):
            clicked_y[0] = None
        elif key == ord('q'):
            break

    cv2.destroyWindow(WIN)

    if thresh is not None and save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_calib(save_path, thresh, W, H, fps)

    return thresh


def run(source, det_model_path, player_height_m=1.70,
        calib_frames=90, result_sec=3.0, conf_thr=0.10,
        save=False, infer_size=640,
        save_calib_path=None, load_calib_path=None):

    print("모델 로딩 중...")
    pose_model = YOLO("yolov8n-pose.pt")
    det_model  = YOLO(det_model_path)
    print(f"셔틀콕 모델: {det_model_path}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")

    # 먼저 native fps 확인 후 필요시에만 해상도 조정
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    if fps_native < 5:  # 캡처보드처럼 낮은 fps일 때만 720p 강제
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("카메라 초기화 중...")
    for _ in range(10):
        cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"해상도: {W}x{H} @ {fps:.0f}fps")
    print("시작! 'q' 키로 종료\n")

    # 저장된 캘리브레이션 불러오기
    preset_thresh = None
    if load_calib_path and Path(load_calib_path).exists():
        data = load_calib(load_calib_path)
        preset_thresh = data["height_thresh_px"]
    elif load_calib_path:
        print(f"[캘리브레이션 파일 없음] {load_calib_path} → 포즈 기반 자동 캘리브레이션 실행")

    # ffmpeg 저장
    ffmpeg_proc = None
    if save:
        ffmpeg_proc = subprocess.Popen(
            ["/opt/homebrew/bin/ffmpeg", "-y",
             "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{W}x{H}", "-pix_fmt", "bgr24",
             "-r", str(fps), "-i", "-",
             "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-crf", "23", "intl_live_result.mp4"],
            stdin=subprocess.PIPE
        )
        print("저장 중: intl_live_result.mp4")

    state = SharedState()
    result_frames = int(result_sec * fps)

    # 저장된 캘리브레이션 불러오기
    if preset_thresh is not None:
        state.height_thresh = preset_thresh
        state.calib_done    = True
        print(f"[캘리브레이션 스킵] 저장된 값 사용: {preset_thresh:.0f}px")

    # 수동 클릭 캘리브레이션 모드
    elif save_calib_path:
        thresh = manual_calib(cap, save_calib_path)
        if thresh is not None:
            state.height_thresh = float(thresh)
            state.calib_done    = True

    # 백그라운드 추론 스레드 시작
    worker = threading.Thread(
        target=inference_worker,
        args=(pose_model, det_model, state,
              calib_frames, conf_thr, infer_size,
              player_height_m, result_frames),
        daemon=True
    )
    worker.start()

    # ── 메인 루프: 카메라 읽기 + 화면 표시 ──────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 추론 스레드에 최신 프레임 전달
        with state.lock:
            state.latest_frame = frame.copy()

        # 현재 감지 결과 읽기 (잠금 최소화)
        with state.lock:
            kps           = state.kps
            sh            = state.shuttle
            height_thresh = state.height_thresh
            calib_done    = state.calib_done
            height_ys_len = len(state.height_ys)
            current_fault = state.current_fault
            fault_cnt     = state.fault_cnt
            ok_cnt        = state.ok_cnt
            sh_hist       = list(state.shuttle_hist)

        # 정지 판단 (표시용)
        recent = [p for p in sh_hist if p is not None]
        stationary = False
        if len(recent) >= 5:
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            stationary = (max(xs)-min(xs)) < 20 and (max(ys)-min(ys)) < 20

        # ── 오버레이 그리기 ─────────────────────────
        draw_skeleton(frame, kps)

        if height_thresh is not None:
            hy = int(height_thresh)
            cv2.line(frame, (0, hy), (W, hy), (0, 200, 255), 2)
            put_ko(frame, "1.15m", (8, hy - 30), size=22, color=(0, 200, 255))
        else:
            put_ko(frame, f"캘리브레이션 중... {height_ys_len}/{calib_frames}",
                   (10, 30), size=22, color=(255, 200, 0))

        if sh:
            sx, sy = int(sh[0]), int(sh[1])
            sy1    = int(sh[4])
            is_above = height_thresh is not None and sy1 < height_thresh
            color    = (0, 60, 255) if is_above else (0, 255, 80)
            cv2.circle(frame, (sx, sy), 16, color, 2)
            cv2.circle(frame, (sx, sy1), 5, color, -1)
            if stationary:
                cv2.circle(frame, (sx, sy), 22, (255, 255, 0), 2)

        draw_result_box(frame, current_fault, fault_cnt, ok_cnt)

        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(frame.tobytes())

        cv2.imshow("Intl Live Detector  |  q: 종료", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    state.running = False

    # 캘리브레이션 결과 저장
    if save_calib_path and state.height_thresh is not None:
        save_calib(save_calib_path, state.height_thresh, W, H, fps)

    cap.release()
    cv2.destroyAllWindows()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print("\n저장 완료: intl_live_result.mp4")

    print(f"\n결과: 폴트 {state.fault_cnt}개 / 정상 {state.ok_cnt}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="국제시합 1.15m 라이브 검출기")
    parser.add_argument("--source",        default="0")
    parser.add_argument("--det_model",     default="best_shuttle_only.pt")
    parser.add_argument("--player_height", type=float, default=1.70)
    parser.add_argument("--calib_frames",  type=int,   default=90)
    parser.add_argument("--result_sec",    type=float, default=3.0)
    parser.add_argument("--conf",          type=float, default=0.10)
    parser.add_argument("--infer_size",    type=int,   default=640)
    parser.add_argument("--side",          default="right", choices=["right","left"])
    parser.add_argument("--save",          action="store_true")
    parser.add_argument("--save_calib",    default=None,           help="캘리브레이션 결과 저장 경로 (예: calib_A코트.json)")
    parser.add_argument("--load_calib",    default="calib_코트.json", help="저장된 캘리브레이션 불러오기 (기본: calib_코트.json)")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    run(src, args.det_model, args.player_height,
        args.calib_frames, args.result_sec, args.conf,
        args.save, args.infer_size,
        save_calib_path=args.save_calib,
        load_calib_path=args.load_calib)
