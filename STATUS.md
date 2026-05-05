# 배드민턴 서브 폴트 검출기 — 현재 상태 (2026-05-05)

## 다른 컴퓨터에서 시작하는 법

```bash
git clone https://github.com/minjunge55/badminton-serve-fault.git
cd badminton-serve-fault
pip install ultralytics opencv-python pillow
```

**별도로 필요한 파일 (GitHub에 없음 — Google Drive에서 다운):**
- `best_v10.pt` → Google Drive / `best_v10.pt`
- `best_v11.pt` → v11 학습 완료 후 Drive에서 다운 (아직 학습 중)

---

## ✅ 완료된 것

### 모델
| 버전 | 상태 | 비고 |
|------|------|------|
| best_v6.pt | ✅ 완료 | Roboflow+우리데이터 혼합, 최초 기준선 |
| best_v9.pt | ✅ 완료 | 90도 측면 데이터만 3,029장 |
| **best_v10.pt** | ✅ 완료 | **현재 최고 성능** racket_head 95.7% |
| best_v11.pt | 🔄 학습 예정 | dataset_v11.zip 준비 완료 |

### 코드 개선
- 웨이스트 폴트: 셔틀콕 상단(y1) 기준으로 수정 (전체가 허리 아래)
- 샤프트 폴트: 서브 시작~임팩트 전체 구간 체크
- 결과 영상: 항상 `.mp4` 저장
- 라이브 검출기: 허리선 즉시 감지 + 30프레임마다 갱신
- 라이브 검출기: `--save` 옵션으로 영상 저장
- 서브 시작: proximity 기반 정밀화 (셔틀+라켓 거리 기반)
- 쉐이크 구간: 최소 2초(120프레임) 보장으로 확장

### 라벨링 데이터
- dataset_v10: 3,971장 (racket_head 2,406개)
- dataset_v11: 4,258장 (racket_head 2,765개) ← v11 학습용
- ZIP 위치: `E:\서브폴트개발\badminton_dataset_v11.zip`

---

## ❌ 미해결 문제

### 1. 라켓헤드 감지율 ~50%
**원인:** 90도 측면 라켓헤드 라벨링 데이터 부족
**해결책:** v11 학습 후 개선 예상 (racket_head 2,765개)
**다음:** v11 학습 → 테스트

### 2. 쉐이크 폴트 미감지
**원인:** 서브 시작점이 임팩트에 너무 가깝게 잡혀 감지 구간이 좁음
**현재 작업:** 최소 2초 구간 보장 + 서브 시작 로직 개선 (인서서브폴트 테스트 진행 중)
**확인 필요:** 테스트 결과 보고 추가 조정

### 3. 샤프트 폴트 0개
**원인:** 라켓헤드가 임팩트 순간에 잘 감지 안 됨
**해결책:** 라켓헤드 감지율 향상 (v11) 후 자연히 개선 예상

### 4. 서브 시작/끝 정확도
**현재:** 손목 속도 기반 (부정확)
**목표:** 셔틀콕 ↔ 라켓헤드 proximity 기반으로 전환
**진행도:** 서브 시작은 proximity 정밀화 완료, 임팩트는 손목속도 유지

### 5. 캡처카드 미구비
**현황:** 마이크로HDMI→USB 캡처카드 구매 예정
**연결:** FDR-AX40 → 마이크로HDMI → 캡처카드 → USB-C(변환젠더) → 맥북
**완료되면:** `python live_detector.py --source 0 --det_model best_v11.pt`

### 6. 허리선 동적 조정 미완
**현황:** 90프레임 보정 후 매 30프레임 갱신 (라이브 기준)
**문제:** 영상 검출기는 처음 90프레임 고정 → 다리 굽힐 때 허리선 내려가야 함
**해결책:** 매 프레임 동적 업데이트 (계산비용 증가)

---

## 다음 할 작업 순서

### 즉시 (지금 진행 중)
- [ ] 인서서브폴트 쉐이크 개선 테스트 결과 확인
- [ ] 결과 보고 추가 로직 조정

### v11 학습
```python
# Colab에서 실행
!pip install ultralytics -q

from google.colab import drive
drive.mount('/content/drive')

import zipfile, yaml, shutil, os
from ultralytics import YOLO

with zipfile.ZipFile('/content/drive/MyDrive/badminton_dataset_v11.zip') as z:
    z.extractall('/content/')

with open('/content/dataset_v11/data.yaml') as f:
    data = yaml.safe_load(f)
data['train'] = '/content/dataset_v11/train/images'
data['val']   = '/content/dataset_v11/valid/images'
with open('/content/data.yaml', 'w') as f:
    yaml.dump(data, f)

def save_checkpoint(trainer):
    if (trainer.epoch + 1) % 10 == 0:
        try:
            src = '/content/runs/badminton_v11/weights/last.pt'
            shutil.copy(src, f'/content/drive/MyDrive/v11_epoch{trainer.epoch+1}.pt')
            print(f'✅ epoch {trainer.epoch+1} 저장')
        except Exception as e:
            print(f'저장 실패: {e}')

model = YOLO('/content/drive/MyDrive/best_v10.pt')
model.add_callback('on_train_epoch_end', save_checkpoint)
model.train(
    data='/content/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='badminton_v11',
    device=0,
    project='/content/runs'
)

shutil.copy('/content/runs/badminton_v11/weights/best.pt',
            '/content/drive/MyDrive/best_v11.pt')
print('완료!')
```

### 추가 라벨링 필요 항목
| 영상 | 추가 목표 | 포인트 |
|------|---------|--------|
| 민지서브폴트.mov | 라켓헤드 100장+ | 90도 측면 서브 준비 |
| 형우서브폴트.mov | 라켓헤드 100장+ | 90도 측면 임팩트 |
| 쉐이크폴트.MTS | 라켓헤드 100장+ | 흔드는 순간 |
| 선밟기.mp4 | service_line 50장+ | 라인 선명한 프레임 |

---

## 폴트별 현재 감지 상태

| 폴트 | 감지 여부 | 정확도 | 비고 |
|------|----------|--------|------|
| 웨이스트 | ✅ | 보통 | 셔틀콕 상단 기준 |
| 높이 1.15m | ✅ | 보통 | 서브 시작 시점 |
| 샤프트 | ⚠️ | 낮음 | 라켓헤드 감지율 부족 |
| 쉐이크 | ⚠️ | 낮음 | 구간 확장 개선 중 |
| 발이동 | ✅ | 보통 | 손목 노이즈 영향 |
| 선밟기 | ❌ | 미구현 | service_line 감지 개선 필요 |
| 헛치기 | ⚠️ | 보통 | 오탐 있음 |

---

## 영상 테스트 명령어

```bash
# 영상 폴트 검출
python serve_fault_detector.py 영상.mp4 --det_model best_v11.pt

# 라이브 (웹캠/캡처카드)
python live_detector.py --source 0 --det_model best_v11.pt

# 라이브 + 영상 저장
python live_detector.py --source 0 --det_model best_v11.pt --save

# 영상 파일로 라이브 테스트
python live_detector.py --source 영상.mp4 --det_model best_v11.pt
```

---

## 카메라 연결 (FDR-AX40)

```
[FDR-AX40] → 마이크로HDMI → [캡처카드 HDMI IN]
[캡처카드 USB-A OUT] → [USB-A to C 변환젠더] → [맥북 M2]
```

- 캡처카드: 마이크로HDMI→USB 타입 (1~2만원)
- 젠더: USB-A to USB-C (애플 정품 있음)
- 연결 후 `--source 0` 또는 `--source 1` 시도
