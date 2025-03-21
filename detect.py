import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import threading
from PIL import Image, ImageDraw, ImageFont

# Tải mô hình đã huấn luyện
model = load_model("D:/AI_IOT/HandGestureRecognition/hand_emotion_model.keras")
emotions = ['Bực bội', 'Buồn', 'Đói', 'Ghen tỵ', 'Hứng thú', 
            'Không thích', 'Lo lắng', 'Tức giận', 'Vui', 'Xấu hổ']

# Khởi tạo MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5)

# Biến lưu cảm xúc trước đó
previous_emotion = None
speech_thread = None  # Luồng chạy âm thanh

# Biến đếm số khung hình không có hành động
frame_count = 0
max_no_action_frames = 15  # Sau 15 khung hình không có cử động -> Reset về "Đang nhận diện..."
previous_data = None  # Lưu dữ liệu khung hình trước đó

# Khai báo biến cảm xúc mặc định
emotion = "Đang nhận diện..."

# Mở webcam
cap = cv2.VideoCapture(0)

def play_audio(text):
    """Phát âm thanh bằng gTTS chạy song song, tránh lag."""
    tts = gTTS(text, lang="vi")
    tts.save("emotion.mp3")
    os.system("start emotion.mp3")

def draw_text_vietnamese(image, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    """Vẽ chữ tiếng Việt bằng PIL để tránh lỗi font."""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # Giảm kích thước ảnh để tăng tốc xử lý
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    valid_data_detected = False
    data = []

    # Nếu phát hiện đủ dữ liệu (cả 2 tay và cánh tay)
    if results.right_hand_landmarks and results.left_hand_landmarks and results.pose_landmarks:
        # Lấy dữ liệu bàn tay phải
        for landmark in results.right_hand_landmarks.landmark:
            data.extend([landmark.x, landmark.y, landmark.z])

        # Lấy dữ liệu bàn tay trái
        for landmark in results.left_hand_landmarks.landmark:
            data.extend([landmark.x, landmark.y, landmark.z])

        # Lấy dữ liệu cánh tay
        arm_points = [mp_holistic.PoseLandmark.RIGHT_SHOULDER, 
                      mp_holistic.PoseLandmark.RIGHT_ELBOW, 
                      mp_holistic.PoseLandmark.RIGHT_WRIST, 
                      mp_holistic.PoseLandmark.LEFT_SHOULDER, 
                      mp_holistic.PoseLandmark.LEFT_ELBOW, 
                      mp_holistic.PoseLandmark.LEFT_WRIST]

        for point in arm_points:
            landmark = results.pose_landmarks.landmark[point]
            data.extend([landmark.x, landmark.y, landmark.z])

        # Chuyển dữ liệu thành numpy array
        data = np.array(data)

        # Kiểm tra sự thay đổi của dữ liệu (người có đang cử động không?)
        if previous_data is not None:
            diff = np.linalg.norm(data - previous_data)  # Tính khoảng cách giữa hai khung hình

            if diff < 0.001:  # Nếu sự thay đổi quá nhỏ, coi như đứng yên
                frame_count += 1
            else:
                frame_count = 0  # Reset bộ đếm nếu có cử động

        previous_data = data  # Cập nhật dữ liệu của khung hình trước

        # Nếu dữ liệu hợp lệ và có thay đổi đáng kể thì dự đoán
        if np.sum(data) > 0 and np.std(data) > 0.01 and frame_count == 0:
            valid_data_detected = True
            data = data.reshape(1, -1)
            prediction = model.predict(data)
            emotion = emotions[np.argmax(prediction)]

            # Chỉ phát âm thanh nếu cảm xúc thay đổi
            if emotion != previous_emotion:
                previous_emotion = emotion
                if speech_thread is None or not speech_thread.is_alive():
                    speech_thread = threading.Thread(target=play_audio, args=(f"Bạn đang cảm thấy {emotion}",))
                    speech_thread.start()

    # Nếu số khung hình không hoạt động vượt quá giới hạn, reset về "Đang nhận diện..."
    if frame_count >= max_no_action_frames:
        emotion = "Đang nhận diện..."
        previous_emotion = None  # Reset cảm xúc trước đó để sẵn sàng nhận diện mới

    # Hiển thị chữ cảm xúc lên màn hình bằng PIL (hỗ trợ tiếng Việt)
    frame = draw_text_vietnamese(frame, f"Cảm xúc: {emotion}", (50, 50), font_size=48)

    # Hiển thị camera với chữ cảm xúc
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
