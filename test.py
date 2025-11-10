import sounddevice as sd
import numpy as np

FS = 44100

def get_noise(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print("Volume:", volume_norm)

with sd.InputStream(callback=get_noise, channels=1, samplerate=FS):
    sd.sleep(5000)  # record for 5 seconds
