import librosa
import numpy as np
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset"
SAMPLE_RATE = 16000

# ── EKSTRAKSI FITUR ──
print("=== TAHAP 1: Ekstraksi Fitur ===")
X = []
y = []

for kata in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, kata)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std  = np.std(mfcc, axis=1)

            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

            fitur = np.concatenate([mfcc_mean, mfcc_std, chroma_mean])
            X.append(fitur)
            y.append(kata.lower())
            print(f"  ✅ {kata}/{file} selesai diekstrak")

X = np.array(X, dtype=np.float64)
y = np.array(y)
print(f"\nTotal data: {len(X)} sampel, {len(set(y))} kata\n")

# ── NORMALISASI ──
print("=== TAHAP 2: Normalisasi ===")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("  ✅ Normalisasi selesai!")

# ── TRAINING ──
print("\n=== TAHAP 3: Training Model ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.001
)
model.fit(X_train, y_train)
print("  ✅ Training selesai!")

# ── EVALUASI ──
print("\n=== TAHAP 4: Evaluasi ===")
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"  Akurasi: {akurasi * 100:.1f}%")

# ── SIMPAN MODEL ──
print("\n=== TAHAP 5: Simpan Model ===")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  ✅ Model tersimpan sebagai model.pkl")
print("  ✅ Scaler tersimpan sebagai scaler.pkl")
print("\nSelesai! Siap lanjut ke GUI.")