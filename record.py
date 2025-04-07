import os 
import sounddevice as sd
from scipy.io.wavfile import write
COMMANDS = ["hey_macbook", "orda_misin", "nasilsin"]
RECORD_TIME = 2
SAMPLING_SPEED = 44100

def make_dir():
    for command in COMMANDS:
        os.makedirs(f"datas/{command}", exist_ok=True)

def sound_record(command_name, index):
    print(f"{command_name} için ses kaydediliyor...({index})")
    record  = sd.rec(int(RECORD_TIME * SAMPLING_SPEED),samplerate=SAMPLING_SPEED, channels=1)
    sd.wait()
    file_path = f"datas/{command_name}/{command_name}_{index}.wav"
    write(file_path,SAMPLING_SPEED,record)
    print(f"{file_path} kaydedildi.")

if __name__ == "__main__":
        make_dir()
        again = int(input("Her komut için kaç kayıt alalım? "))
        
        for command in COMMANDS:
            input(f"{command} komutu için hazır mısın? Enter'a bas.")
            for i in range(again):
                sound_record(command, i+1)    