# 배드민턴 서브 폴트 자동 검출 시스템

실시간 영상에서 BWF 공식 규정 기반으로 배드민턴 서브 폴트를 자동 감지하는 시스템

---

## 주요 기능

- **1.15m 높이 폴트 감지** — 마우스로 기준선 설정, 셔틀콕 위치 자동 판정
- **실시간 라이브 감지** — 캠코더 → 캡처보드 → 맥북 파이프라인
- **폴트 시 음성 알림** — "fault" 음성 출력 (Flo, 미국 여성 목소리)
- **이동 중 오감지 방지** — 셔틀콕 정지 상태일 때만 판정 (서브 중 이동 시 무시)
- **시각적 피드백** — 폴트: 빨간 원 / 정상: 초록 원 / 이동 중: 회색 원

---

## 하드웨어 구성

```
소니 AX캠 (캠코더)
    ↓ HDMI
MBeat 캡처보드 (USB3.0 Capture)
    ↓ USB
맥북
    ↓
live_detector.py 실행
```

- **카메라 0** = MBeat 캡처보드 (AX캠 연결 시)
- **카메라 1** = 맥북 내장 FaceTime (사용 안 함)

---

## 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
ultralytics>=8.0
opencv-python>=4.8
numpy>=1.24
Pillow
```

---

## 실행 방법

### 라이브 (체육관)
```bash
python live_detector.py \
  --source 0 \
  --det_model /Users/parkminjeong/Documents/work/badminton-serve-fault/best_shuttle_only.pt
```

### 영상 파일 테스트
```bash
python live_detector.py \
  --source 영상파일.mp4 \
  --det_model best_shuttle_only.pt
```

### 옵션
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `0` | 카메라 번호 또는 영상 파일 경로 |
| `--det_model` | `best_v9.pt` | YOLO 감지 모델 경로 |
| `--skip` | `1` | N프레임마다 YOLO 처리 (높을수록 빠름) |
| `--infer_size` | `320` | YOLO 추론 해상도 (낮을수록 빠름) |

---

## 사용법 (현장)

1. AX캠 HDMI → MBeat → 맥북 연결
2. `live_detector.py` 실행
3. 화면에서 **마우스 클릭/드래그**로 1.15m 기준선 위치 조절
4. **Enter** 키로 기준선 확정
5. 셔틀콕 감지 시작

| 키 | 기능 |
|---|---|
| 마우스 클릭/드래그 | 1.15m 기준선 이동 |
| Enter | 기준선 확정 |
| r | 기준선 재설정 |
| f | 전체화면 토글 |
| q | 종료 |

---

## 시각적 표시

| 색상 | 의미 |
|---|---|
| 회색 원 | 셔틀콕 이동 중 (판정 없음) |
| 초록 원 | 정지 + 기준선 아래 (정상) |
| 빨간 원 | 정지 + 기준선 위 (폴트) → "fault" 음성 출력 |
| 주황 가로선 | 1.15m 기준선 (확정됨) |
| 노란 가로선 | 1.15m 기준선 (조절 중) |

---

## 카메라 탐색

```bash
python find_camera.py
```

연결된 카메라 목록과 미리보기 확인

---

## 모델

| 파일 | 클래스 | 설명 |
|---|---|---|
| `best_shuttle_only.pt` | shuttlecock | 셔틀콕 전용 (현재 라이브 사용) |
| `best_v9.pt` | shuttlecock, racket_head, service_line | 통합 모델 v9 |
| `best_v11.pt` | shuttlecock, racket_head, service_line | 통합 모델 v11 (최신) |

> 모델 파일(`.pt`)은 용량 문제로 일부만 저장됨. 전체 모델은 로컬 `/Documents/work/badminton-serve-fault/` 참고

---

## 학습 데이터

| 클래스 | 프레임 수 | 출처 |
|---|---|---|
| shuttlecock | ~7,000장 | 자체 촬영 + Roboflow |
| racket_head | ~11,000장 | 자체 촬영 |
| service_line | ~1,800장 | 자체 촬영 |
| **합계** | **약 20,000장** | |

**모델 학습 이력**

| 버전 | Epoch | 주요 개선 |
|---|---|---|
| v1 | 60 | Roboflow 셔틀콕 기본 감지 |
| v2 | 60 | 라켓헤드·서비스라인 클래스 추가 |
| v3 | 60 | 자체 영상 391장 추가 |
| v6 | 80 | 헛치기 오탐 로직 개선 |
| v9 | 80+ | 실전 사용 모델 (현재) |
| v11 | 80+ | 최신 통합 모델 |

---

## BWF 규정 대응 폴트 목록

| 폴트 | BWF 조항 | 감지 방식 | 구현 |
|---|---|---|---|
| 높이 폴트 | 9.1.6.2 | 셔틀콕 위치 vs 1.15m 기준선 | ✅ 라이브 |
| 웨이스트 폴트 | 9.1.6 | 셔틀콕 위치 vs 허리선 | ✅ serve_fault_detector |
| 샤프트 폴트 | 9.1.7 | 라켓헤드 vs 손목 위치 | ✅ serve_fault_detector |
| 쉐이크 폴트 | 9.1.8 | 라켓헤드 x방향 속도 | ✅ serve_fault_detector |
| 빽지체 폴트 | 9.1.8 | 라켓헤드 이동 방향 전환 | ✅ serve_fault_detector |
| 선밟기 폴트 | 9.1.4 | 발목 vs 서비스라인 | ✅ serve_fault_detector |

---

## 파일 구조

```
badminton-serve-fault/
├── live_detector.py       # 라이브 실시간 감지 (메인)
├── serve_fault_detector.py # 영상 분석용 전체 폴트 감지
├── find_camera.py         # 카메라 탐색 유틸리티
├── extract_frames.py      # 학습용 프레임 추출
├── train_model.py         # YOLO 모델 학습
├── best_v9.pt             # 학습된 모델 (v9)
├── yolov8n-pose.pt        # 포즈 추정 모델
├── PLAN.md                # 개발 계획 및 BWF 규정 매핑
└── requirements.txt       # 패키지 목록
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 창이 바로 꺼짐 | 프로세스 충돌 | `pkill -f live_detector` 후 재실행 |
| No Signal | HDMI 빠짐 | AX캠 → MBeat HDMI 재연결 |
| 맥북 화면만 나옴 | 카메라 번호 틀림 | `find_camera.py`로 번호 확인 후 `--source` 변경 |
| 선이 안 보임 | Enter 안 누름 | 마우스 클릭으로 선 위치 조절 후 Enter |
| 소리 안 남 | 맥북 음소거 | 시스템 볼륨 확인 |
| 감지 속도 느림 | 추론 해상도 높음 | `--skip 2 --infer_size 320` 추가 |
