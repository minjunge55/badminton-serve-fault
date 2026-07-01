"""
기존 데이터셋에서 셔틀콕(class 0)만 추출해 단일 클래스 훈련용 ZIP 생성.

사용법:
  python make_shuttle_dataset.py badminton_dataset_v9.zip
  → shuttle_dataset.zip + shuttle_data.yaml 생성
"""

import zipfile
import argparse
import shutil
from pathlib import Path


def filter_label(label_text):
    """라벨 파일에서 class 0(셔틀콕) 행만 남기고 클래스 번호를 0으로 유지."""
    kept = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) == 0:
            kept.append(line)
    return "\n".join(kept)


def make_dataset(src_zip_path, out_zip_path="shuttle_dataset.zip",
                 yaml_path="shuttle_data.yaml", tmp_dir="/tmp/shuttle_ds"):

    src_zip_path = Path(src_zip_path)
    tmp = Path(tmp_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"[1/3] {src_zip_path.name} 압축 해제 중...")
    with zipfile.ZipFile(src_zip_path) as z:
        z.extractall(tmp)

    # 루트 폴더 찾기 (dataset_v9/ 같은 이름)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"예상치 못한 ZIP 구조: {roots}")
    root = roots[0]

    splits = [p.name for p in root.iterdir() if p.is_dir() and p.name in ("train", "valid", "test")]
    print(f"  분할: {splits}")

    out_tmp = tmp / "shuttle_dataset"
    kept_total = 0
    skipped_total = 0

    print("[2/3] 셔틀콕 라벨만 필터링 중...")
    for split in splits:
        img_src  = root / split / "images"
        lbl_src  = root / split / "labels"
        img_dst  = out_tmp / split / "images"
        lbl_dst  = out_tmp / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        label_files = list(lbl_src.glob("*.txt")) if lbl_src.exists() else []
        for lbl_file in label_files:
            filtered = filter_label(lbl_file.read_text(encoding="utf-8"))
            if not filtered.strip():
                skipped_total += 1
                continue  # 셔틀콕 라벨 없는 이미지는 제외

            # 대응 이미지 찾기 (.jpg / .png)
            stem = lbl_file.stem
            img_file = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidate = img_src / (stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                skipped_total += 1
                continue

            (lbl_dst / lbl_file.name).write_text(filtered, encoding="utf-8")
            shutil.copy2(img_file, img_dst / img_file.name)
            kept_total += 1

        print(f"  {split}: {kept_total}개 유지 (현재까지)")

    print(f"\n  유지: {kept_total}개 / 제외(셔틀콕 없음): {skipped_total}개")

    # data.yaml 생성
    yaml_content = f"""path: /content/shuttle_dataset
train: train/images
val:   valid/images

nc: 1
names:
  0: shuttlecock
"""
    yaml_dst = out_tmp / "data.yaml"
    yaml_dst.write_text(yaml_content, encoding="utf-8")
    Path(yaml_path).write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml 저장: {yaml_path}")

    print(f"\n[3/3] {out_zip_path} 압축 중...")
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for f in out_tmp.rglob("*"):
            if f.is_file():
                zout.write(f, "shuttle_dataset" / f.relative_to(out_tmp))

    shutil.rmtree(tmp)
    size_mb = Path(out_zip_path).stat().st_size / 1024 / 1024
    print(f"\n완료: {out_zip_path} ({size_mb:.1f} MB)")
    print("→ Google Drive에 업로드 후 Colab에서 훈련하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="셔틀콕 전용 데이터셋 생성")
    parser.add_argument("src_zip",              help="기존 데이터셋 ZIP (예: badminton_dataset_v9.zip)")
    parser.add_argument("-o", "--out",          default="shuttle_dataset.zip")
    parser.add_argument("--yaml",               default="shuttle_data.yaml")
    parser.add_argument("--tmp",                default="/tmp/shuttle_ds")
    args = parser.parse_args()

    make_dataset(args.src_zip, args.out, args.yaml, args.tmp)
