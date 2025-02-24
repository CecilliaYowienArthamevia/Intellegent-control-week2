import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')

# Periksa apakah dataset memiliki kolom yang benar
if not {'R', 'G', 'B', 'ColorName'}.issubset(color_data.columns):
    raise ValueError("Dataset harus memiliki kolom 'R', 'G', 'B', dan 'ColorName'.")

X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Gunakan MinMaxScaler untuk menjaga proporsi warna
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

detected_colors_1 = []
detected_colors_2 = []
detected_true_labels_1 = []
detected_true_labels_2 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi BGR ke RGB agar sesuai dengan dataset
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = frame.shape
    
    # Tentukan dua area bounding box untuk deteksi warna
    box_size = 50
    x1, y1 = width // 3 - box_size // 2, height // 2 - box_size // 2
    x2, y2 = 2 * width // 3 - box_size // 2, height // 2 - box_size // 2
    
    roi1 = frame_rgb[y1:y1+box_size, x1:x1+box_size]
    roi2 = frame_rgb[y2:y2+box_size, x2:x2+box_size]
    
    # Ambil rata-rata warna dalam bounding box
    pixel1 = np.mean(roi1, axis=(0, 1))
    pixel2 = np.mean(roi2, axis=(0, 1))
    
    # Normalisasi nilai pixel agar sesuai dengan model
    pixel1_scaled = scaler.transform(pixel1.reshape(1, -1))
    pixel2_scaled = scaler.transform(pixel2.reshape(1, -1))
    
    # Prediksi warna
    color_pred_1 = svm_model.predict(pixel1_scaled)[0]
    color_pred_2 = svm_model.predict(pixel2_scaled)[0]
    
    # Temukan warna asli terdekat (setelah normalisasi)
    distances1 = np.linalg.norm(X_scaled - pixel1_scaled, axis=1)
    distances2 = np.linalg.norm(X_scaled - pixel2_scaled, axis=1)
    true_color_1 = y[np.argmin(distances1)]
    true_color_2 = y[np.argmin(distances2)]
    
    # Simpan prediksi dan warna asli
    detected_colors_1.append(color_pred_1)
    detected_colors_2.append(color_pred_2)
    detected_true_labels_1.append(true_color_1)
    detected_true_labels_2.append(true_color_2)
    
    # Batasi jumlah data untuk perhitungan akurasi real-time
    if len(detected_colors_1) > 50:
        detected_colors_1.pop(0)
        detected_colors_2.pop(0)
        detected_true_labels_1.pop(0)
        detected_true_labels_2.pop(0)
    
    # Hitung akurasi real-time
    if detected_colors_1:
        realtime_accuracy_1 = accuracy_score(detected_true_labels_1, detected_colors_1) * 100
        realtime_accuracy_2 = accuracy_score(detected_true_labels_2, detected_colors_2) * 100
    else:
        realtime_accuracy_1 = 0.0
        realtime_accuracy_2 = 0.0
    
    # Tampilkan prediksi dan akurasi real-time
    cv2.putText(frame, f'Color1: {color_pred_1} | Acc1: {realtime_accuracy_1:.2f}%', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Color2: {color_pred_2} | Acc2: {realtime_accuracy_2:.2f}%', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Gambar bounding box pada frame dengan keterangan
    cv2.rectangle(frame, (x1, y1), (x1+box_size, y1+box_size), (255, 0, 0), 2)
    cv2.rectangle(frame, (x2, y2), (x2+box_size, y2+box_size), (0, 0, 255), 2)
    cv2.putText(frame, "Color 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Color 2", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    print(f'Color1: {color_pred_1} | Acc1: {realtime_accuracy_1:.2f}%')
    print(f'Color2: {color_pred_2} | Acc2: {realtime_accuracy_2:.2f}%')
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
