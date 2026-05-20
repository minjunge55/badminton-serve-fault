"""
카메라 탐색 유틸리티
사용법: python find_camera.py
"""

import cv2
import sys

def find_cameras(max_index=8):
    print("카메라 탐색 중...\n")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            status = "OK" if ret else "열림(프레임 없음)"
            print(f"  카메라 {i}: {int(w)}x{int(h)} @ {fps:.0f}fps  [{status}]")
            if ret:
                found.append(i)
        cap.release()
    print()
    return found

def preview_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print(f"카메라 {idx} 열기 실패")
        return
    print(f"카메라 {idx} 미리보기 — q 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Camera {idx}  (q:quit)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(f"Camera {idx} Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    found = find_cameras()
    if not found:
        print("사용 가능한 카메라 없음")
        sys.exit(1)

    print(f"사용 가능한 카메라: {found}")
    if len(found) == 1:
        idx = found[0]
    else:
        try:
            idx = int(input(f"미리보기할 카메라 번호 입력 {found}: "))
        except (ValueError, EOFError):
            idx = found[0]

    preview_camera(idx)
