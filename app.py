from flask import Flask, render_template, request, jsonify, send_file
import sounddevice as sd
import librosa
import numpy as np
import pickle
import os
import threading
from gtts import gTTS

app = Flask(__name__)

# ── LOAD MODEL ──
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── DATABASE INFO ALAT MUSIK ──
info_alat_musik = {
    "gitar"   : "Gitar adalah alat musik petik modern yang memiliki senar dan sering digunakan dalam berbagai genre musik seperti pop, rock, dan jazz.",
    "piano"   : "Piano adalah alat musik modern yang dimainkan menggunakan keyboard dan menghasilkan nada dari senar di dalamnya.",
    "drums"   : "Drums adalah alat musik pukul modern yang digunakan untuk mengatur ritme dalam musik.",
    "terompet": "Terompet adalah alat musik tiup modern berbahan logam yang menghasilkan suara nyaring dan sering digunakan dalam musik jazz dan orkestra.",
    "biola"   : "Biola adalah alat musik gesek modern yang dimainkan menggunakan penggesek atau bow dan menghasilkan suara yang merdu.",
    "suling"  : "Suling adalah alat musik tiup tradisional yang biasanya terbuat dari bambu dan banyak digunakan dalam musik daerah Indonesia.",
    "kendang" : "Kendang adalah alat musik pukul tradisional yang digunakan dalam musik daerah Indonesia seperti gamelan dan dangdut.",
    "gong"    : "Gong adalah alat musik pukul tradisional berbentuk lingkaran logam yang menghasilkan suara khas dan dalam.",
    "gamelan" : "Gamelan adalah ansambel musik tradisional Indonesia yang terdiri dari berbagai alat musik pukul dan petik.",
    "angklung": "Angklung adalah alat musik tradisional dari Jawa Barat yang dimainkan dengan cara digoyangkan dan telah diakui UNESCO."
}

SAMPLE_RATE = 16000
DURASI = 2

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/rekam", methods=["POST"])
def rekam():
    try:
        # Rekam suara
        audio = sd.rec(int(DURASI * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1, dtype='float32')
        sd.wait()
        audio_data = audio.flatten()

        # Ekstrak fitur
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc_mean   = np.mean(mfcc, axis=1)
        mfcc_std    = np.std(mfcc, axis=1)
        chroma      = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE)
        chroma_mean = np.mean(chroma, axis=1)
        fitur = np.concatenate([mfcc_mean, mfcc_std, chroma_mean]).reshape(1, -1)
        fitur = scaler.transform(fitur.astype(np.float64))

        # Prediksi
        prediksi     = model.predict(fitur)[0]
        probabilitas = model.predict_proba(fitur)[0]
        confidence   = round(max(probabilitas) * 100, 1)
        info         = info_alat_musik.get(prediksi, "Informasi tidak tersedia.")

        # Simpan MFCC sebagai gambar
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title("Visualisasi MFCC")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Koefisien")
        plt.tight_layout()
        plt.savefig("static/mfcc.png")
        plt.close()

        return jsonify({
            "status"    : "success",
            "prediksi"  : prediksi.upper(),
            "confidence": confidence,
            "info"      : info
        })
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)})

@app.route("/tts", methods=["POST"])
def tts():
    try:
        data      = request.json
        teks      = data.get("teks", "")
        kecepatan = data.get("kecepatan", "normal")
        slow      = True if kecepatan == "lambat" else False

        tts_obj = gTTS(text=teks, lang='id', slow=slow)
        tts_obj.save("static/output_tts.mp3")

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "pesan": str(e)})

@app.route("/simpan")
def simpan():
    path = "static/output_tts.mp3"
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="hasil_tts.mp3")
    return jsonify({"status": "error", "pesan": "File belum ada"})

if __name__ == "__main__":
    app.run(debug=True)