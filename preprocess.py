import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Khởi tạo MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Thư mục chứa video
DATASET_PATH = "D:/AI_IOT/Camxuc"
OUTPUT_PATH = "D:/AI_IOT/HandGestureRecognition/extracted_data"

# Danh sách cảm xúc
emotions = ['buc_boi', 'buon', 'doi', 'ghen_ty', 'hung_thu', 'khong_thich', 'lo_lang', 'tuc_gian', 'vui', 'xau_ho']

# Tạo thư mục lưu dữ liệu đã trích xuất
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Xử lý từng video
for emotion in emotions:
    emotion_path = os.path.join(DATASET_PATH, emotion)
    output_csv = os.path.join(OUTPUT_PATH, f"{emotion}.csv")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame"] + 
                        [f"x_rh{i}" for i in range(21)] + [f"y_rh{i}" for i in range(21)] + [f"z_rh{i}" for i in range(21)] +  
                        [f"x_lh{i}" for i in range(21)] + [f"y_lh{i}" for i in range(21)] + [f"z_lh{i}" for i in range(21)] +  
                        [f"x_a{i}" for i in range(6)] + [f"y_a{i}" for i in range(6)] + [f"z_a{i}" for i in range(6)] +  
                        ["label"])

        for video_name in os.listdir(emotion_path):
            video_path = os.path.join(emotion_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                # Vẽ landmark lên khung hình
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Chuyển lại BGR để hiển thị
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Hiển thị video
                cv2.imshow("Video", frame)

                # Thoát nếu nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Trích xuất dữ liệu
                row = [frame_num]

                # Bàn tay phải
                if results.right_hand_landmarks:
                    for landmark in results.right_hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                else:
                    row.extend([0] * 63)

                # Bàn tay trái
                if results.left_hand_landmarks:
                    for landmark in results.left_hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z])
                else:
                    row.extend([0] * 63)
                
                # Cánh tay
                if results.pose_landmarks:
                    arm_points = [mp_holistic.PoseLandmark.RIGHT_SHOULDER, 
                                  mp_holistic.PoseLandmark.RIGHT_ELBOW, 
                                  mp_holistic.PoseLandmark.RIGHT_WRIST, 
                                  mp_holistic.PoseLandmark.LEFT_SHOULDER, 
                                  mp_holistic.PoseLandmark.LEFT_ELBOW, 
                                  mp_holistic.PoseLandmark.LEFT_WRIST]
                    
                    for point in arm_points:
                        landmark = results.pose_landmarks.landmark[point]
                        row.extend([landmark.x, landmark.y, landmark.z])
                else:
                    row.extend([0] * 18)

                row.append(emotion)
                writer.writerow(row)

                frame_num += 1

            cap.release()
            cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị khi xong từng video

print("✅ Hoàn thành trích xuất dữ liệu!")
