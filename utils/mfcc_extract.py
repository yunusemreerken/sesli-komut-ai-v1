import os
import librosa
import numpy as np
import pickle

COMMANDS = ["hey_macbook","orda_misin","nasilsin"]
DATA_PATH = "datas/"
MFCC_PATH = "datas/mfcc/"
SAMPLE_SPEED = 44100
N_MFCC = 13 # classic MFCC dimensiol

def create_mfcc():
    os.makedirs(MFCC_PATH,exist_ok=True)
    datas = []
    labels = []

    for index, command in enumerate(COMMANDS):
        command_directory = os.path.join(DATA_PATH,command)
        files = os.listdir(command_directory)

        for file in files:
            file_path = os.path.join(command_directory,file)
            try:
                sound, _  = librosa.load(file_path, sr=SAMPLE_SPEED)
                mfcc = librosa.feature.mfcc(y=sound, sr =SAMPLE_SPEED,n_mfcc = N_MFCC)
                mfcc = np.mean(mfcc.T,axis=0)
                datas.append(mfcc)
                labels.append(index)
            except Exception as e:
                print(f"Error: {file_path} - {e}")
    
    with open(os.path.join(MFCC_PATH,"data.pickle"),"wb") as f:
        pickle.dump((np.array(datas), np.array(labels)),f)
    
    print("MFCC çıkarımı tamamlandı.")


if __name__ == "__main__":
    create_mfcc()
