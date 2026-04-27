"""
영상에서 라벨링용 프레임 추출
사용법: python extract_frames.py
"""

import cv2
import os
from pathlib import Path

VIDEO_DIR = Path(__file__).parent  # badminton-serve-fault 폴더
VIDEO_DIR_PARENT = Path(__file__).parent.parent  # 서브폴트개발 폴더
OUTPUT_DIR = Path(__file__).parent / "frames_for_labeling"

# (파일명, 폴더) 형태로 지정
VIDEOS = [
    ("민지서브폴트.mov",  VIDEO_DIR),
    ("형우서브폴트.mov",  VIDEO_DIR),
    ("인서서브폴트.mov",  VIDEO_DIR),
    ("정상.mp4",          VIDEO_DIR_PARENT),
    ("허리선넘기.mp4",    VIDEO_DIR_PARENT),
    ("쉐이크.mp4",        VIDEO_DIR_PARENT),
    ("오버헤드.mp4",      VIDEO_DIR_PARENT),
    ("빽지체.mp4",        VIDEO_DIR_PARENT),
    ("풋무브.mp4",        VIDEO_DIR_PARENT),
    ("선밟기.mp4",        VIDEO_DIR_PARENT),
    ("쉐이크폴트.MTS",   VIDEO_DIR_PARENT),
    ("앞45도.MTS",        VIDEO_DIR_PARENT),  # 앞 45도 각도 영상
    ("뒤45도.MTS",        VIDEO_DIR_PARENT),  # 뒤 45도 각도 영상
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
    for name, folder in VIDEOS:
        vpath = folder / name
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
