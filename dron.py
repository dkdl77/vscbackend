import os
import cv2
import numpy as np
import torch
import shutil
from sklearn.cluster import KMeans

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# 영상 불러오기
video_path = "c.mp4"
cap = cv2.VideoCapture(video_path)

# 저장 디렉토리 초기화
save_dir = "detected_people"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# HSV 색상 범위 (예: 검정색 옷)
target_upper_lower_hsv = {
    "upper": [
    (np.array([0, 100, 100]), np.array([10, 255, 255]))
],
    "lower": [
    (np.array([0, 100, 100]), np.array([10, 255, 255]))
]
}

# 피부색 범위 (HSV)
skin_lower = np.array([0, 30, 60], dtype=np.uint8)
skin_upper = np.array([20, 150, 255], dtype=np.uint8)

# YOLO 탐지 설정
MIN_PERSON_HEIGHT = 100
CONF_THRESHOLD = 0.5

# 피부색 제거 후 유효한 BGR 픽셀 반환
def get_valid_pixels(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    mask_inv = cv2.bitwise_not(skin_mask)
    return image[mask_inv > 0]

# HSV 범위와 일치 여부 판단
def is_color_in_range(image, hsv_ranges):
    valid_pixels = get_valid_pixels(image)
    if len(valid_pixels) == 0:
        return False
    hsv_pixels = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mask = np.zeros(len(hsv_pixels), dtype=bool)
    for lower, upper in hsv_ranges:
        in_range = np.all((hsv_pixels >= lower) & (hsv_pixels <= upper), axis=1)
        mask |= in_range
    ratio = np.sum(mask) / len(mask)
    return ratio > 0.4

# 주요 색상 추출
def extract_dominant_colors(image, k=3):
    valid_pixels = get_valid_pixels(image)
    if len(valid_pixels) < k:
        return []
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(valid_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    return centers

# 프레임 설정
frame_count = 0
frame_interval = 30  # 1초 간격 (30fps 기준)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()

    for i, (*box, conf, cls) in enumerate(detections):
        if int(cls) != 0 or conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        if (y2 - y1) < MIN_PERSON_HEIGHT:
            continue

        # 전체 높이
        full_height = y2 - y1
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int(full_height * 0.1)

        # 상체: 상단 마진 포함, 45%
        upper_y1 = y1 + margin_y
        upper_y2 = y1 + int(full_height * 0.45)
        upper_img = frame[upper_y1:upper_y2, x1 + margin_x:x2 - margin_x]

        # 하체: 하단 마진 포함, 55%
        lower_y1 = y1 + int(full_height * 0.45)
        lower_y2 = y2 - margin_y
        lower_img = frame[lower_y1:lower_y2, x1 + margin_x:x2 - margin_x]

        is_upper_match = is_color_in_range(upper_img, target_upper_lower_hsv["upper"])
        is_lower_match = is_color_in_range(lower_img, target_upper_lower_hsv["lower"])

        if is_upper_match and is_lower_match:
            # 원래 바운딩 박스로 전체 이미지 저장
            crop_img = frame[y1:y2, x1:x2]
            filename = f"{save_dir}/person_{frame_count}_{i}.jpg"
            cv2.imwrite(filename, crop_img)
            print(f"✅ 저장됨: {filename}")

            # 주요 색상 출력
            dominant_colors_upper = extract_dominant_colors(upper_img)
            dominant_colors_lower = extract_dominant_colors(lower_img)
            print(f"👕 상의 주요 색상: {dominant_colors_upper}")
            print(f"👖 하의 주요 색상: {dominant_colors_lower}")

            # 디버깅용 박스 시각화
            cv2.rectangle(frame, (x1 + margin_x, upper_y1), (x2 - margin_x, upper_y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1 + margin_x, lower_y1), (x2 - margin_x, lower_y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imshow("people", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()