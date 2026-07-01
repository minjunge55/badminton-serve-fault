# 국제시합 라이브 서브 폴트 감지기

## 장비 연결
```
소니 AX캠 HDMI OUT
        ↓
MBeat 캡처보드 HDMI IN
        ↓ USB
Apple 멀티포트 어댑터 USB-A
        ↓ USB-C
맥북
```

## 카메라 번호 확인
```bash
cd ~/Documents/work/badminton-serve-fault
venv/bin/python -c "
import cv2
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'카메라 {i}: {w}x{h}')
    cap.release()
" 2>/dev/null
```
> 소니 AX캠은 보통 **1번** (1920x1080)

## 라이브 실행

### 기본 (오른손 서브)
```bash
cd ~/Documents/work/badminton-serve-fault
venv/bin/python intl_live_detector.py --source 1 --side right
```

### 왼손 서브
```bash
venv/bin/python intl_live_detector.py --source 1 --side left
```

### 영상 저장하면서
```bash
venv/bin/python intl_live_detector.py --source 1 --side right --save
```

### 느릴 때 (처리 빠르게)
```bash
venv/bin/python intl_live_detector.py --source 1 --side right --skip 3 --infer_size 480
```

## 종료
화면에서 **`q`** 키

## 화면 설명
| 표시 | 의미 |
|------|------|
| 노란 가로선 | 1.15m 기준선 |
| 초록 원 | 셔틀콕 (기준선 아래 — 정상) |
| 빨간 원 | 셔틀콕 (기준선 위 — 폴트 가능) |
| 노란 테두리 원 | 정지 셔틀콕 (판정 대상) |
| 우측 하단 박스 | 폴트/정상 누적 카운트 |
| 스켈레톤 | 포즈 감지 (1.15m 캘리브레이션용) |

## 캘리브레이션 주의사항
- 시작 후 **첫 90프레임** 동안 화면에 보이는 사람 기준으로 자동 캘리브레이션
- **실제 서브할 선수**가 카메라 정면에 서있는 상태에서 실행해야 정확함
- 캘리브레이션 완료 전에는 1.15m 선이 표시되지 않음
