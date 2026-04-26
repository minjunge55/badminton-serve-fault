"""
영상에서 라벨링용 프레임 추출
사용법: python extract_frames.py
"""

import cv2
import os
from pathlib import Path

VIDEO_DIR = Path(__file__).parent  # 영상 파일과 같은 폴더
OUTPUT_DIR = Path(__file__).parent / "frames_for_labeling"
VIDEOS = [
    "민지서브폴트.mov",
    # 나중에 추가할 영상들:
    # "형우서브폴트.mov",
    # "인서서브폴트.mov",
    # "정상.mp4",
    # "허리선넘기.mp4",
    # "선밟기.mp4",
    # "쉐이크.mp4",
    # "풋무브.mp4",
    # "오버헤드.mp4",
    # "빽지체.mp4",
]
EVERY_N_FRAMES = 5  # 5프레임마다 1장 추출 (60fps → 초당 12장)


def extract(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [오류] 열 수 없음: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  {video_path.name}: {total}프레임 / {fps:.0f}fps")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    saved = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % EVERY_N_FRAMES == 0:
            fname = out_dir / f"{stem}_{idx:05d}.jpg"
            # cv2.imwrite는 한글 경로에서 실패 → imencode + 직접 쓰기
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                fname.write_bytes(buf.tobytes())
                saved += 1
        idx += 1

    cap.release()
    return saved


def main():
    print("=== 프레임 추출 시작 ===")
    print(f"출력 폴더: {OUTPUT_DIR}\n")

    total_saved = 0
    for name in VIDEOS:
        vpath = VIDEO_DIR / name
        if not vpath.exists():
            print(f"[스킵] 파일 없음: {vpath}")
            continue
        print(f"처리 중: {name}")
        n = extract(vpath, OUTPUT_DIR / Path(name).stem)
        print(f"  → {n}장 저장\n")
        total_saved += n

    print(f"=== 완료: 총 {total_saved}장 ===")
    print(f"\n다음 단계:")
    print(f"  1. {OUTPUT_DIR} 폴더를 Roboflow에 업로드")
    print(f"  2. 'shuttlecock' 과 'racket_head' 클래스로 바운딩박스 라벨링")
    print(f"  3. YOLOv8 포맷으로 export → train_model.py 실행")


if __name__ == "__main__":
    main()
