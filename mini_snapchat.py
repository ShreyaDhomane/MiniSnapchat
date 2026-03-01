import cv2
import os
import math
import numpy as np
import mediapipe as mp
import time


# Paths

base_path = os.path.dirname(os.path.abspath(__file__))

# Face filters
face_cascade_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
sunglasses_path = os.path.join(base_path, "filters", "sunglasses.png")
mustache_path = os.path.join(base_path, "filters", "mustache.png")
dog_ears_path = os.path.join(base_path, "filters", "dog_ears.png")
heart_path = os.path.join(base_path, "filters", "heart.png")

# Button icons 
icon_sunglasses_path = os.path.join(base_path, "filters", "icons", "sunglasses_icon.png")
icon_mustache_path = os.path.join(base_path, "filters", "icons", "mustache_icon.png")
icon_dog_ears_path = os.path.join(base_path, "filters", "icons", "dog_ears_icon.png")
icon_heart_path = os.path.join(base_path, "filters", "icons", "heart_icon.png")


# Load Face Cascade

face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Face cascade not found!")
    exit()


# Load Filter Images

sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
mustache = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)
dog_ears = cv2.imread(dog_ears_path, cv2.IMREAD_UNCHANGED)
heart_img = cv2.imread(heart_path, cv2.IMREAD_UNCHANGED)

# Load Button Icons
icon_sunglasses = cv2.imread(icon_sunglasses_path, cv2.IMREAD_UNCHANGED)
icon_mustache = cv2.imread(icon_mustache_path, cv2.IMREAD_UNCHANGED)
icon_dog_ears = cv2.imread(icon_dog_ears_path, cv2.IMREAD_UNCHANGED)
icon_heart = cv2.imread(icon_heart_path, cv2.IMREAD_UNCHANGED)

if any(x is None for x in [sunglasses, mustache, dog_ears, heart_img,
                           icon_sunglasses, icon_mustache, icon_dog_ears, icon_heart]):
    print("One or more images not found!")
    exit()

# Resize large images once
sunglasses = cv2.resize(sunglasses, (800, 800))
mustache = cv2.resize(mustache, (800, 800))
dog_ears = cv2.resize(dog_ears, (800, 800))

# Resize icons for buttons
icon_size = 30
icon_sunglasses = cv2.resize(icon_sunglasses, (icon_size, icon_size))
icon_mustache = cv2.resize(icon_mustache, (icon_size, icon_size))
icon_dog_ears = cv2.resize(icon_dog_ears, (icon_size, icon_size))
icon_heart = cv2.resize(icon_heart, (icon_size, icon_size))

# Overlay Function

def overlay_filter(frame, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame
    roi = frame[y1:y2, x1:x2]
    overlay_bgr = overlay_resized[:, :, :3]
    alpha = overlay_resized[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=-1)
    blended = (overlay_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    return frame


# MediaPipe HandLandmarker Setup

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
from mediapipe.tasks.python.vision.core import image as mp_image

model_path = "hand_landmarker.task"  

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
timestamp_ms = 0

# Buttons Setup 

buttons = {
    "Glasses": {"pos": (10, 50, 150, 50), "icon": icon_sunglasses, "filter": 1},
    "Mustache": {"pos": (10, 120, 150, 50), "icon": icon_mustache, "filter": 2},
    "B&W": {"pos": (10, 190, 150, 50), "icon": None, "filter": 3},
    "Dog Ears": {"pos": (10, 260, 150, 50), "icon": icon_dog_ears, "filter": 4},
    "Heart": {"pos": (10, 330, 150, 50), "icon": icon_heart, "filter": 5},
    "Normal": {"pos": (10, 400, 150, 50), "icon": None, "filter": 0}
}

hover_start = {}
hover_duration = 0.5  
filter_mode = 0


# Camera Setup

cap = cv2.VideoCapture(0)


# Main Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # face filters
    for (x, y, w, h) in faces:
        if filter_mode == 1:
            frame = overlay_filter(frame, sunglasses, x, y + h//4, w, h//3)
        elif filter_mode == 2:
            frame = overlay_filter(frame, mustache, x, y + h//2, w, h//3)
        elif filter_mode == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif filter_mode == 4:
            overlay_resized = cv2.resize(dog_ears, (w, h))
            ears_only = overlay_resized[0:int(h*0.5), :, :]
            ear_width = int(w * 1.3)
            ear_height = int(h * 0.5)
            ear_x = x - int(w * 0.15)
            ear_y = y - int(h * 0.35)
            frame = overlay_filter(frame, ears_only, ear_x, ear_y, ear_width, ear_height)
            # Nose
            nose_width = int(w * 0.4)
            nose_height = int(h * 0.3)
            nose_x = x + w//2 - nose_width//2
            nose_y = y + h//2 - nose_height//2
            nose_only = overlay_resized[int(h*0.55):int(h*0.85), int(w*0.3):int(w*0.7)]
            frame = overlay_filter(frame, nose_only, nose_x, nose_y, nose_width, nose_height)

    
    # Hand Detection
    
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    hand_result = hand_landmarker.detect_for_video(mp_img, timestamp_ms)
    timestamp_ms += 1

    finger_x = finger_y = None
    if hand_result.hand_landmarks:
        hand = hand_result.hand_landmarks[0]
        index_tip = hand[8]
        thumb_tip = hand[4]

        finger_x = int(index_tip.x * frame.shape[1])
        finger_y = int(index_tip.y * frame.shape[0])

        # Hand Heart detection
        if filter_mode == 5:
            x1 = int(thumb_tip.x * frame.shape[1])
            y1 = int(thumb_tip.y * frame.shape[0])
            x2 = int(index_tip.x * frame.shape[1])
            y2 = int(index_tip.y * frame.shape[0])
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < 0.15 * frame.shape[1]:
                heart_size = 120
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)
                hover_offset = 40
                heart_x = mid_x - heart_size // 2
                heart_y = mid_y - heart_size - hover_offset
                frame = overlay_filter(frame, heart_img, heart_x, heart_y, heart_size, heart_size)

    
    # Buttons with Hover Animation & Icons
    
    current_time = time.time()
    for name, info in buttons.items():
        bx, by, bw, bh = info["pos"]
        icon = info["icon"]

        color = (50, 50, 50)  # default gray
        scale = 1.0

        # Hover effect
        if finger_x is not None and bx <= finger_x <= bx+bw and by <= finger_y <= by+bh:
            color = (0, 120, 255)  # highlight
            scale = 1.1  
            if name not in hover_start:
                hover_start[name] = current_time
            elif current_time - hover_start[name] >= hover_duration:
                filter_mode = info["filter"]
        else:
            hover_start.pop(name, None)

        # button rectangle with hover scale
        bw_scaled = int(bw * scale)
        bh_scaled = int(bh * scale)
        bx_scaled = bx - (bw_scaled - bw)//2
        by_scaled = by - (bh_scaled - bh)//2
        cv2.rectangle(frame, (bx_scaled, by_scaled), (bx_scaled+bw_scaled, by_scaled+bh_scaled), color, -1)

        # button text
        cv2.putText(frame, name, (bx_scaled+10, by_scaled+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # icon if exists
        if icon is not None:
            icon_h, icon_w = icon.shape[:2]
            icon_x = bx_scaled + bw_scaled - icon_w - 10
            icon_y = by_scaled + (bh_scaled - icon_h)//2
            frame = overlay_filter(frame, icon, icon_x, icon_y, icon_w, icon_h)

    cv2.imshow("Mini Snapchat", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
