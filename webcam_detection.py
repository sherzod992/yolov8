"""
YOLOv8 + CNN 실시간 객체 탐지 및 분류 프로그램

시스템 흐름:
1) 웹캠에서 실시간 영상 프레임을 입력받는다
2) YOLOv8이 화면 전체를 보고 객체의 위치, 종류, 확신도를 출력한다
3) YOLO가 탐지한 Bounding Box 좌표로 원본 프레임에서 해당 영역만 잘라낸다(crop)
4) 잘라낸 이미지를 CNN의 입력으로 사용한다
5) CNN이 해당 이미지가 정말 사람인지 재확인한다 (person / not-person 이진 분류)
6) YOLO 결과와 CNN 판단 결과를 함께 출력한다

"YOLO는 빠른 감시자, CNN은 정밀 검사관"
- YOLO: 전체 화면을 빠르게 스캔하여 "어디에 무엇이 있는지" 찾아냄
- CNN: YOLO가 찾은 영역을 자세히 보고 "정확히 무엇인지" 판단함 (사람 재확인)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os

print("YOLOv8 모델을 로드하는 중...")
yolo_model = YOLO('yolov8n.pt')

print("CNN 모델을 로드하는 중...")
MODEL_PATH = 'person_classifier.pth'

if os.path.exists(MODEL_PATH):
    base_model = models.resnet18(weights=None)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, 2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.to(device)
    base_model.eval()
    cnn_model = base_model
    print(f"학습된 모델을 로드했습니다: {MODEL_PATH}")
    print("CNN 역할: 사람 재확인 (person / not-person 이진 분류)")
else:
    print(f"경고: 학습된 모델을 찾을 수 없습니다 ({MODEL_PATH})")
    print("ImageNet 기반 모델을 사용합니다 (일반 객체 분류)")
    print("사람 재확인 기능을 사용하려면 train_person_classifier.py를 실행하여 모델을 학습하세요.")
    cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    cnn_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
    exit()

print("웹캠 객체 탐지를 시작합니다. ESC 키를 누르면 종료됩니다.")
print("=" * 60)
print("시스템 설명:")
print("- YOLO: 전체 화면을 빠르게 스캔하여 객체를 찾습니다 (빠른 감시자)")
print("- CNN: YOLO가 찾은 영역을 자세히 분석합니다 (정밀 검사관)")
if os.path.exists(MODEL_PATH):
    print("- CNN 역할: 사람 재확인 (person / not-person)")
else:
    print("- CNN 역할: 일반 객체 분류 (ImageNet 기반)")
print("=" * 60)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    results = yolo_model(frame, verbose=False)
    
    annotated_frame = results[0].plot()
    
    detections = results[0].boxes
    
    person_count = 0
    processed_persons = set()
    
    for i, box in enumerate(detections):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        class_name = yolo_model.names[class_id]
        
        if class_name == 'person' and confidence >= 0.5:
            person_count += 1
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            person_id = f"{x1}_{y1}_{x2}_{y2}"
            
            if person_id not in processed_persons:
                print(f"[YOLO 탐지] Person #{person_count}: YOLO가 사람을 탐지했습니다 (확신도: {confidence*100:.1f}%)")
                processed_persons.add(person_id)
            
            frame_height, frame_width = frame.shape[:2]
            x1 = max(0, min(x1, frame_width))
            y1 = max(0, min(y1, frame_height))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            cropped_image = frame[y1:y2, x1:x2]
            
            if cropped_image.size == 0 or cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                continue
            
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
            input_tensor = transform(cropped_rgb)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            print(f"[CNN 확인 중] Person #{person_count}: CNN이 확인 중입니다...")
            
            with torch.no_grad():
                cnn_output = cnn_model(input_batch)
            
            if os.path.exists(MODEL_PATH):
                probabilities = torch.nn.functional.softmax(cnn_output[0], dim=0)
                person_prob = probabilities[0].item()
                not_person_prob = probabilities[1].item()
                
                is_person = person_prob > 0.5
                cnn_confidence = person_prob if is_person else not_person_prob
                
                if is_person:
                    print(f"[CNN 확인 완료] Person #{person_count}: 사람이다! (확신도: {person_prob*100:.1f}%)")
                else:
                    print(f"[CNN 확인 완료] Person #{person_count}: 사람이 아니다 (확신도: {not_person_prob*100:.1f}%)")
            else:
                probabilities = torch.nn.functional.softmax(cnn_output[0], dim=0)
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                is_person = None
                cnn_confidence = None
                print(f"[CNN 확인 완료] Person #{person_count}: ImageNet 기반 분류 완료 (상위 클래스: {top5_indices[0].item()}, 확률: {top5_prob[0].item()*100:.1f}%)")
            
            display_crop = cropped_image.copy()
            if display_crop.shape[0] < 200 or display_crop.shape[1] < 200:
                scale = 200 / min(display_crop.shape[0], display_crop.shape[1])
                new_width = int(display_crop.shape[1] * scale)
                new_height = int(display_crop.shape[0] * scale)
                display_crop = cv2.resize(display_crop, (new_width, new_height))
            
            y_offset = 20
            cv2.putText(display_crop, f"Person #{person_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            if os.path.exists(MODEL_PATH):
                color = (0, 255, 0) if is_person else (0, 0, 255)
                status = "Person" if is_person else "Not Person"
                cv2.putText(display_crop, f"CNN: {status} ({cnn_confidence*100:.1f}%)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
                cv2.putText(display_crop, f"Person: {person_prob*100:.1f}%", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                cv2.putText(display_crop, f"Not Person: {not_person_prob*100:.1f}%", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                for idx in range(min(3, len(top5_indices))):
                    class_idx = top5_indices[idx].item()
                    prob = top5_prob[idx].item()
                    
                    class_label = f"CNN Top {idx+1}: Class {class_idx} ({prob*100:.1f}%)"
                    cv2.putText(display_crop, class_label, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20
            
            cv2.imshow(f'Crop Person #{person_count} (CNN Input)', display_crop)
            
            if os.path.exists(MODEL_PATH):
                info_text = f"Person #{person_count}: YOLO={confidence:.2f}, CNN={'Person' if is_person else 'Not Person'}"
                color = (0, 255, 0) if is_person else (0, 0, 255)
            else:
                info_text = f"Person #{person_count}: YOLO={confidence:.2f}"
                color = (0, 255, 0)
            
            cv2.putText(annotated_frame, info_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow('YOLOv8 실시간 객체 탐지 (YOLO → Crop → CNN)', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")
