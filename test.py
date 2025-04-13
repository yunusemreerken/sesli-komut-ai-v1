import sounddevice as sd
import librosa
import numpy as np
import pickle
from model.my_ann import MyANN

KOMUTLAR = ["hey_macbook", "orda_misin", "nasilsin"]
SAMPLE_RATE = 44100
SURE = 2  # saniye

def kayit_al(sure=SURE, sr=SAMPLE_RATE):
    print("🎙️ Komut söyle, kayıt başlıyor...")
    ses = sd.rec(int(sure * sr), samplerate=sr, channels=1)
    sd.wait()
    return ses.flatten()

def mfcc_hazirla(ses, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def sesli_cevap(veri):
    from pyttsx3 import init
    motor = init()
    motor.say(veri)
    motor.runAndWait()

def test_et():
    ses = kayit_al()
    mfcc = mfcc_hazirla(ses)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # normalize

    model = MyANN()
    model.load("model/ann_weights.pkl")

    tahmin = model.predict(mfcc)
    komut = KOMUTLAR[tahmin]

    print(f"🧠 Tahmin edilen komut: {komut}")

    # Sesli cevap
    if komut == "hey_macbook":
        sesli_cevap("efendim ")
    elif komut == "orda_misin":
        sesli_cevap("Evet, buradayım")
    elif komut == "nasilsin":
        sesli_cevap("Harikayım, sen nasılsın?")

if __name__ == "__main__":
    test_et()
