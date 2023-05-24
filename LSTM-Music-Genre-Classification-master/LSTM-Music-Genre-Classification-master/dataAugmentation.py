import librosa
import numpy as np
import random
import itertools
import IPython.display as ipd
import matplotlib.pyplot as plt


def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data



def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()
    
    
original_audio = "./gtzan/train/classical.au"

rate = 0.9
augmented_audio = librosa.effects.time_stretch(original_audio, rate = rate)


au = load_audio_file(original_audio)
print("hola")

