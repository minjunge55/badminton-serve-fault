"""
배드민턴 커스텀 YOLO 학습
클래스 0: shuttlecock
클래스 1: racket_head

사용법:
  python train_model.py dataset/data.yaml
  python train_model.py dataset/data.yaml --epochs 150 --batch 32
  python train_model.py dataset/data.yaml --val --model runs/train/badminton_v1/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def train(data_yaml, epochs=100, imgsz=640, batch=16, device="0", name="badminton_v1"):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=name,
        project="runs/train",
        patience=20,
        save=True,
        plots=True,
    )
    best = f"runs/train/{name}/weights/best.pt"
    print(f"\n학습 완료. 최적 모델: {best}")
    return best


def validate(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"\nmAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    for i, name in enumerate(["shuttlecock", "racket_head"]):
        print(f"  {name}: mAP50={metrics.box.maps[i]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="배드민턴 YOLO 커스텀 학습")
    parser.add_argument("data", help="Roboflow export data.yaml 경로")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU 번호 또는 'cpu'")
    parser.add_argument("--name",   default="badminton_v1", help="실험 이름")
    parser.add_argument("--val",    action="store_true", help="학습 없이 검증만 실행")
    parser.add_argument("--model",  default=None, help="--val 시 사용할 모델 경로")
    args = parser.parse_args()

    if args.val:
        if not args.model:
            parser.error("--val 사용 시 --model 경로를 지정하세요.")
        validate(args.model, args.data)
    else:
        train(args.data, args.epochs, args.imgsz, args.batch, args.device, args.name)
