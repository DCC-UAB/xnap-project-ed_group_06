import librosa
import numpy as np
import random
import itertools
import IPython.display as ipd
import matplotlib.pyplot as plt
import soundfile as sf


def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def load_audio_file_no_limit(file_path):
    data, _ = librosa.core.load(file_path, sr=None)
    return data



def plot_time_series(data,name):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.savefig('data_aug_{}'.format(name))
    plt.show()
    

original_audio = "./gtzan/_train/classical.00030.au"

#careguem àudio
au = load_audio_file_no_limit(original_audio)
rate = 0.75 #velocitat àudio

#plot àudio original
plot_time_series(au, 'original')

#data augmentation àudio --> soroll
augmented_audio = librosa.effects.time_stretch(au, rate = rate)
augmented_audio = augmented_audio[:661794]

#plot àudio soroll
plot_time_series(augmented_audio, 'time')

signal_rate = 22050  # Tasa de muestreo de audio
print(original_audio)

#path nou àudio
original_audio = original_audio.split('/')[-1].split('.')
output_file = './gtzan/_train/'+str(original_audio[0])+'.'+str(original_audio[1])+'_time.'+str(original_audio[2])
print(output_file)

#es carrega nou àudio
sf.write(output_file, augmented_audio, signal_rate)

#-----------------------NOISE-----------------------------
wn = np.random.randn(len(au))
augmented_audio2 = au + 0.0075*wn

#plot àudio noise
plot_time_series(augmented_audio2, 'noise')

#path nou àudio
output_file = './gtzan/_train/'+str(original_audio[0])+'.'+str(original_audio[1])+'_noise.'+str(original_audio[2])
print(output_file)

#es carrega nou àudio
sf.write(output_file, augmented_audio2, signal_rate)



