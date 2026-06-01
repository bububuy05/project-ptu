import os
from pydub import AudioSegment

DATASET_PATH = "dataset"

print("=== KONVERSI AUDIO ===\n")

total_berhasil = 0
total_gagal = 0

for kata in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, kata)
    if not os.path.isdir(folder):
        continue
    
    print(f"📁 Konversi folder: {kata.upper()}")
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        
        if file.endswith(".wav") or file.endswith(".m4a") or file.endswith(".mp3") or file.endswith(".ogg"):
            try:
                audio = AudioSegment.from_file(path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                
                # Simpan ulang sebagai WAV
                new_path = os.path.splitext(path)[0] + ".wav"
                audio.export(new_path, format="wav")
                print(f"  ✅ {file} berhasil dikonversi")
                total_berhasil += 1
            except Exception as e:
                print(f"  ❌ {file} gagal: {e}")
                total_gagal += 1

print(f"\n=== SELESAI ===")
print(f"✅ Berhasil : {total_berhasil} file")
print(f"❌ Gagal    : {total_gagal} file")
print("\nSekarang jalankan training.py!")