import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')

# Pastikan dataset memiliki kolom yang sesuai
if not {'R', 'G', 'B', 'ColorName'}.issubset(color_data.columns):
    raise ValueError("Dataset harus memiliki kolom 'R', 'G', 'B', dan 'ColorName'.")

X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Gunakan MinMaxScaler untuk mempertahankan proporsi warna
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training Model ML dengan SVM menggunakan kernel RBF
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Prediksi data test
y_pred = svm_model.predict(X_test)

# Menghitung akurasi model awal
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model Awal: {accuracy * 100:.2f}%")

cap = cv2.VideoCapture(0)

detected_colors = []
detected_true_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame_rgb[height // 2, width // 2]
    pixel_center_reshaped = pixel_center.reshape(1, -1)
    pixel_center_scaled = scaler.transform(pixel_center_reshaped)

    # Prediksi warna
    color_pred = svm_model.predict(pixel_center_scaled)[0]

    # Temukan warna asli terdekat dari dataset
    distances = np.linalg.norm(X - pixel_center, axis=1)
    nearest_idx = np.argmin(distances)
    true_color = y[nearest_idx]

    # Simpan prediksi dan warna asli
    detected_colors.append(color_pred)
    detected_true_labels.append(true_color)

    # Batasi jumlah data untuk perhitungan akurasi real-time
    if len(detected_colors) > 50:
        detected_colors.pop(0)
        detected_true_labels.pop(0)

    # Hitung akurasi real-time
    if detected_colors:
        realtime_accuracy = accuracy_score(detected_true_labels, detected_colors) * 100
    else:
        realtime_accuracy = 0.0

    # Tampilkan prediksi dan akurasi real-time
    cv2.putText(frame, f'Color: {color_pred} | Accuracy: {realtime_accuracy:.2f}%', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f'Color: {color_pred} | Accuracy: {realtime_accuracy:.2f}%')

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
