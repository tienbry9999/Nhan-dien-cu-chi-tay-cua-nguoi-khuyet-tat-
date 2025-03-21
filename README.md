🖐 Hệ thống nhận diện cử chỉ tay trong thời gian thực và chuyển những ký hiệu thành giọng nói 🎤

![image](https://github.com/user-attachments/assets/8c916761-15c4-4aef-a7df-65a8f0197355)


🚀 Giới thiệu
Dự án này sử dụng MediaPipe, TensorFlow, OpenCV để nhận diện cử chỉ tay và chuyển đổi chúng thành giọng nói thông qua webcam. Hệ thống áp dụng Mạng Neural Nhân tạo (ANN) với kiến trúc Sequential, được huấn luyện để nhận diện 10 cử chỉ tay khác nhau, giúp hỗ trợ giao tiếp cho người khiếm thính hoặc trong các tình huống đặc biệt.
🎯 Tính năng chính
✅ Nhận diện cử chỉ tay trong thời gian thực.
✅ Chuyển đổi cử chỉ thành giọng nói để giao tiếp.
✅ Hỗ trợ 10 cử chỉ tay khác nhau.
✅ Ứng dụng công nghệ AI tiên tiến giúp nhận diện nhanh chóng và chính xác.

📥 Cài đặt
🛠 Điều kiện tiên quyết
🐍 Python 3.7+ – Ngôn ngữ lập trình chính
💾 RAM 8GB+ – Được khuyến nghị để có hiệu suất tối ưu
🖥 CPU 4+ cores – Để xử lý nhanh hơn
📷 Webcam – Để nhận diện cử chỉ tay
🔊 Loa – Để phát giọng nói
🎥 Thiết lập dự án
1️⃣ Clone dự án

sh
Copy code
https://github.com/tienbry9999/Nhan-dien-cu-chi-tay-cua-nguoi-khuyet-tat-.git
cd Hand-Gesture-to-Speech  
🎥 Hướng dẫn sử dụng hệ thống nhận diện cử chỉ tay và chuyển đổi thành giọng nói
1️⃣ Chuẩn bị môi trường làm việc
Đảm bảo máy tính đã cài đặt Python 3.7+.
Cài đặt các thư viện cần thiết bằng lệnh sau:
sh
Copy code
pip install opencv-python mediapipe numpy tensorflow pandas scikit-learn matplotlib pyautogui pyttsx3
2️⃣ Thu thập dữ liệu cử chỉ tay
Sử dụng camera máy tính để quay lại video của từng cử chỉ tay và lưu vào thư mục cam_xuc.
Trong quá trình này:
✅ Bạn cần thực hiện 10 cử chỉ tay tương ứng với các cảm xúc:
Bực bội, buồn, đói, ghen tỵ, hứng thú, không thích, lo lắng, tức giận, vui, xấu hổ
✅ Mỗi video sẽ được gán nhãn cảm xúc tương ứng để phục vụ quá trình huấn luyện mô hình.
✅ Video của từng cử chỉ sẽ được tự động lưu vào thư mục D:/AI_IOT/cam_xuc.
✅ Đảm bảo webcam hoạt động tốt, thực hiện cử chỉ rõ ràng, đúng vị trí để hệ thống nhận diện chính xác.
3️⃣ Trích xuất keypoints từ video và lưu vào file CSV
Sau khi thu thập video, hệ thống sẽ sử dụng Mediapipe để xác định landmark trên bàn tay.
Các điểm landmark này sẽ được ghi lại vào file .csv trong thư mục extracted_data.
Chạy lệnh sau để thực hiện quá trình này:
sh
Copy code
python preprocess.py  
Quá trình này bao gồm:
✅ Đọc từng video trong thư mục cam_xuc.
✅ Xác định keypoints của bàn tay bằng Mediapipe Holistic.
✅ Lưu thông tin landmark vào các file CSV tương ứng trong thư mục extracted_data.
4️⃣ Huấn luyện mô hình nhận diện cử chỉ tay
Sau khi có dữ liệu từ file CSV, tiến hành huấn luyện mô hình bằng lệnh sau:
sh
Copy code
python train.py  
Hệ thống sẽ:
✅ Đọc dữ liệu landmark từ các file CSV.
✅ Sử dụng Mạng Neural Nhân tạo (ANN) với kiến trúc Sequential để huấn luyện mô hình.
✅ Lưu mô hình huấn luyện được dưới dạng hand_emotion_model.keras.
5️⃣ Nhận diện cử chỉ tay và chuyển đổi thành giọng nói
Sau khi huấn luyện xong, có thể chạy chương trình nhận diện và chuyển thành giọng nói bằng lệnh sau:
sh
Copy code
python detect.py  
Hệ thống sẽ:
✅ Sử dụng webcam để quét cử chỉ tay trong thời gian thực.
✅ Nhận diện cảm xúc tương ứng dựa trên mô hình đã huấn luyện.
✅ Chuyển đổi cử chỉ thành giọng nói bằng thư viện pyttsx3.
🛑 Lưu ý
Nhấn 'q' để thoát chương trình nhận diện.
Đảm bảo môi trường đủ ánh sáng để nhận diện chính xác.
Nếu kết quả chưa tốt, có thể thu thập thêm dữ liệu và huấn luyện lại mô hình.
🖐 Các cử chỉ hỗ trợ
✅ 😠 Bực bội – "Bạn đang cảm thấy bực bội."
✅ 😞 Buồn – "Bạn đang cảm thấy buồn."
✅ 🍽 Đói – "Bạn đang cảm thấy đói."
✅ 😒 Ghen tỵ – "Bạn đang cảm thấy ghen tỵ."
✅ 🤩 Hứng thú – "Bạn đang cảm thấy hứng thú."
✅ 👎 Không thích – "Bạn đang cảm thấy không thích."
✅ 😟 Lo lắng – "Bạn đang cảm thấy lo lắng."
✅ 😡 Tức giận – "Bạn đang cảm thấy tức giận."
✅ 😀 Vui – "Bạn đang cảm thấy vui."
✅ 😳 Xấu hổ – "Bạn đang cảm thấy xấu hổ."

📌 Ghi chú
Nhấn 'q' để thoát chương trình.
Đảm bảo webcam và loa hoạt động bình thường để có trải nghiệm tốt nhất.
📝 Giấy phép
© 2025 Nhóm 4 - Lớp CNTT 1603 🎓
🏫 Trường Đại học Đại Nam

