import pickle
import numpy as np
from model.my_ann import MyANN
import os

def train():
    with open("datas/mfcc/data.pickle","rb") as f:
        X,Y = pickle.load(f)

    print(f"🎧 Eğitim Verisi {X.shape}, Etiketler: {Y.shape}")

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    model = MyANN(input_size = 13, hidden_size = 32, output_size = 3, learning_rate = 0.01)

    model.train(X,Y,epochs=300)

    os.makedirs("model", exist_ok=True)
    model.save("model/ann_weights.pkl")
    print("✅ Eğitim tamamlandı ve model kaydedildi.")

if __name__ == "__main__":
    train()