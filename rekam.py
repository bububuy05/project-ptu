import librosa
import numpy as np
import os

DATASET_PATH = "dataset"
SAMPLE_RATE = 16000

print("=== CEK DATASET ===\n")

total_file = 0
for kata in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, kata)
    if not os.path.isdir(folder):
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    total_file += len(files)
    print(f"📁 {kata.upper()} → {len(files)} file")

print(f"\nTotal keseluruhan: {total_file} file audio")
print(f"Total kata: {len(os.listdir(DATASET_PATH))} kata")

if total_file >= 200:
    print("\n✅ Dataset sudah cukup! Siap lanjut training.")
else:
    print("\n⚠️ Dataset kurang! Minimal 200 file (10 kata × 20 sampel)")