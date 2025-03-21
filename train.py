import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

# Thư mục dữ liệu
DATA_PATH = "D:/AI_IOT/HandGestureRecognition/extracted_data"

# Load dữ liệu
data = []
labels = []
emotions = ['buc_boi', 'buon', 'doi', 'ghen_ty', 'hung_thu', 'khong_thich', 'lo_lang', 'tuc_gian', 'vui', 'xau_ho']

for emotion in emotions:
    df = pd.read_csv(os.path.join(DATA_PATH, f"{emotion}.csv"))
    data.append(df.iloc[:, 1:-1].values)  # Lấy các cột toạ độ (bỏ cột frame và label)
    labels.append([emotions.index(emotion)] * len(df))

X = np.vstack(data)
y = np.hstack(labels)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chuyển nhãn về dạng One-Hot Encoding
y = to_categorical(y, num_classes=10)

# Xáo trộn dữ liệu để tránh bias
X, y = shuffle(X, y, random_state=42)

# Xây dựng mô hình chống overfitting
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.2),
    
    Dense(10, activation='softmax')
])

# Compile với Learning Rate thấp hơn từ đầu
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback để chống overfitting
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Huấn luyện mô hình
model.fit(
    X, y, epochs=100, batch_size=64, validation_split=0.2, callbacks=[lr_scheduler, early_stopping]
)

# Lưu mô hình và scaler
model.save("D:/AI_IOT/HandGestureRecognition/hand_emotion_model.keras")
np.save("D:/AI_IOT/HandGestureRecognition/scaler.npy", scaler.mean_)

print("\u2705 Huấn luyện xong!")