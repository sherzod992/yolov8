"""
YOLOv8 웹캠 실시간 객체 탐지 프로그램
노트북 웹캠을 사용하여 실시간으로 객체를 탐지하고 화면에 표시합니다.
"""

import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (pre-trained 모델, 자동으로 다운로드됨)
# 'yolov8n.pt': nano (가장 빠름, 정확도 낮음)
# 'yolov8s.pt': small (균형잡힌 성능)
# 'yolov8m.pt': medium
# 'yolov8l.pt': large
# 'yolov8x.pt': extra large (가장 정확, 느림)
model = YOLO('yolov8n.pt')  # nano 모델 사용 (실시간 처리에 적합)

# 웹캠 초기화 (0번 카메라 = 노트북 내장 웹캠)
cap = cv2.VideoCapture(0)

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    exit()

print("웹캠 객체 탐지를 시작합니다. ESC 키를 누르면 종료됩니다.")

# 실시간 영상 처리 루프
while True:
    # 웹캠에서 한 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임을 읽지 못하면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # YOLOv8로 객체 탐지 수행
    # results: 탐지된 모든 객체들의 정보가 담긴 리스트
    results = model(frame, verbose=False)  # verbose=False: 불필요한 출력 제거
    
    # 탐지 결과를 프레임에 그리기
    annotated_frame = results[0].plot()  # bounding box, label, confidence 자동으로 그려줌
    
    # 화면에 표시
    cv2.imshow('YOLOv8 실시간 객체 탐지', annotated_frame)
    
    # ESC 키(ASCII 코드 27)를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:  # 0xFF는 8비트 마스크
        break

# 리소스 해제
cap.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
print("프로그램을 종료합니다.")
