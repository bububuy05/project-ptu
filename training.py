import os
import pickle
import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# KONFIGURASI
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000

# FUNGSI EKSTRAKSI FITUR
def extract_features(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(delta2, axis=1)

    fitur = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta2_mean
    ])

    fitur = np.nan_to_num(
        fitur,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return fitur


# EKSTRAKSI FITUR + AUGMENTASI
print("=== TAHAP 1: EKSTRAKSI FITUR ===")

X = []
y = []

for kelas in os.listdir(DATASET_PATH):

    folder = os.path.join(
        DATASET_PATH,
        kelas
    )

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        if not file.lower().endswith(".wav"):
            continue

        path = os.path.join(
            folder,
            file
        )

        try:

            audio, sr = librosa.load(
                path,
                sr=SAMPLE_RATE
            )

            if len(audio) == 0:
                continue

            # DATA ASLI
            fitur_asli = extract_features(
                audio,
                sr
            )

            X.append(fitur_asli)
            y.append(kelas.lower())

            # AUGMENTASI NOISE
            noise = np.random.randn(
                len(audio)
            )

            audio_noise = audio + (
                0.003 * noise
            )

            fitur_noise = extract_features(
                audio_noise,
                sr
            )

            X.append(fitur_noise)
            y.append(kelas.lower())

            # AUGMENTASI PITCH SHIFT
            audio_pitch = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=1
            )

            fitur_pitch = extract_features(
                audio_pitch,
                sr
            )

            X.append(fitur_pitch)
            y.append(kelas.lower())

            print(
                f"{kelas}/{file}"
            )

        except Exception as e:

            print(
                f"Error {path}"
            )

            print(e)

# ARRAY
X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

# SHUFFLE
X, y = shuffle(
    X,
    y,
    random_state=42
)

print("\n================================")
print("Shape X :", X.shape)
print("Shape y :", y.shape)
print("Jumlah kelas :", len(np.unique(y)))
print("NaN :", np.isnan(X).sum())
print("Inf :", np.isinf(X).sum())
print("================================")

# NORMALISASI
print("\n=== TAHAP 2: NORMALISASI ===")

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Normalisasi selesai")

# SPLIT DATA
print("\n=== TAHAP 3: SPLIT DATA ===")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train :", len(X_train))
print("Test  :", len(X_test))


# TRAINING MLP
print("\n=== TAHAP 4: TRAINING MLP ===")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.0005,
    max_iter=8000,
    random_state=42,
    verbose=True
)

model.fit(
    X_train,
    y_train
)

train_acc = accuracy_score(
    y_train,
    model.predict(X_train)
)

print(
    f"AKURASI TRAIN : {train_acc*100:.2f}%"
)

print("Training selesai")


# TESTING
print("\nTAHAP 5: TESTING")

y_pred = model.predict(
    X_test
)

akurasi = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAKURASI TEST : {akurasi*100:.2f}%"
)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Prediksi"
)

plt.ylabel(
    "Aktual"
)

plt.tight_layout()
plt.show()

# SIMPAN MODEL
print("\n=== TAHAP 6: SIMPAN MODEL ===")

with open(
    "model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

with open(
    "scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )

print("model.pkl tersimpan")
print("scaler.pkl tersimpan")

print("\nSELESAI")