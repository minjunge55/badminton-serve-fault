"""
셔틀콕 전용 YOLO 모델 학습 스크립트
- 10 에폭마다 Google Drive 자동 저장
- 학습 완료 시 best_shuttle_only.pt 저장

사용법 (Colab):
  !python train_shuttle_only.py
  !python train_shuttle_only.py --epochs 150 --batch 32
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="/content/shuttle_dataset/data.yaml")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--device",     default="0")
    p.add_argument("--save_dir",   default="/content/drive/MyDrive/shuttle_only_checkpoints")
    p.add_argument("--name",       default="shuttle_only_v1")
    p.add_argument("--resume",     default=None, help="이어서 학습할 체크포인트 경로")
    return p.parse_args()


def main():
    args = get_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Drive 저장 경로: {args.save_dir}")

    # ── 자동 저장 콜백 ──────────────────────────────
    def on_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % 10 == 0:
            dst = f"{args.save_dir}/epoch{epoch:03d}.pt"
            shutil.copy(str(trainer.last), dst)
            print(f"\n[자동저장] {dst}")

    def on_train_end(trainer):
        best_dst = f"{args.save_dir}/best_shuttle_only.pt"
        root_dst = str(Path(args.save_dir).parent / "best_shuttle_only.pt")
        shutil.copy(str(trainer.best), best_dst)
        shutil.copy(str(trainer.best), root_dst)
        print(f"\n[완료] best_shuttle_only.pt → Drive 저장")

    # ── 모델 로드 (이어하기 or 새로 시작) ──────────
    if args.resume:
        print(f"이어서 학습: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO("yolov8n.pt")

    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.add_callback("on_train_end",       on_train_end)

    # ── 학습 ────────────────────────────────────────
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="/content/runs",
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        resume=bool(args.resume),
    )

    # ── 검증 ────────────────────────────────────────
    best_path = f"{args.save_dir}/best_shuttle_only.pt"
    print("\n── 검증 ──")
    metrics = YOLO(best_path).val(data=args.data)
    print(f"mAP50:             {metrics.box.map50:.3f}")
    print(f"mAP50-95:          {metrics.box.map:.3f}")
    print(f"shuttlecock mAP50: {metrics.box.maps[0]:.3f}")


if __name__ == "__main__":
    main()
