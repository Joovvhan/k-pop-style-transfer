import pyaudio
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import datetime
from matplotlib.animation import FuncAnimation
import threading
from scipy.io import wavfile
import os
import librosa
from pydub import AudioSegment
from pydub.playback import play
import keyboard
import glob
from scipy import signal
import sounddevice as sd

def keyboard_event_func(e):
    global original_buffer
    global chiptune_buffer
    global piano_buffer
    
    global original_stream
    global chiptune_stream
    global piano_stream
    
#     if e.name == 'delete':
    if e.name == 'p':
        sd.play(original_buffer, fs)
        
    elif e.name == 'l':
        sd.play(chiptune_buffer, fs)
    
    elif e.name == ',':
        sd.play(piano_buffer, fs)
        
    elif e.name == 'q':
        original_buffer = np.append(original_buffer, next(original_stream))
        plot_figure()
        
    elif e.name == 'a':
        chiptune_buffer = np.append(chiptune_buffer, next(chiptune_stream))
        plot_figure()
    
    elif e.name == 'z':
        piano_buffer = np.append(piano_buffer, next(piano_stream))
        plot_figure()
        
    elif e.name == 'w':
        _len = len(original_buffer)
        original_buffer = np.append(original_buffer, next(original_stream))
        original_buffer = original_buffer[-_len:]
        
        plot_figure()
        
    elif e.name == 's':
        _len = len(chiptune_buffer)
        chiptune_buffer = np.append(chiptune_buffer, next(chiptune_stream))
        chiptune_buffer = chiptune_buffer[-_len:]
        
        plot_figure()
    
    elif e.name == 'x':
        _len = len(piano_buffer)
        piano_buffer = np.append(piano_buffer, next(piano_stream))
        piano_buffer = piano_buffer[-_len:]
        
        plot_figure()

        
def plot_figure():
    
    global fig
    global axes
    global plots
    global original_buffer
    global chiptune_buffer
    global piano_buffer
    
    global fs
    
    len_max = np.max([len(original_buffer), len(chiptune_buffer), len(piano_buffer)])
    
#     plots[0][0][0].set_ydata(original_buffer)
    axes[0][0].cla()
    axes[0][0].plot(original_buffer)
    axes[0][0].set_xlim([0, len_max])

#     plots[1][0][0].set_ydata(chiptune_buffer)
    axes[1][0].cla()
    axes[1][0].plot(chiptune_buffer)
    axes[1][0].set_xlim([0, len_max])

#     plots[2][0][0].set_ydata(piano_buffer)
    axes[2][0].cla()
    axes[2][0].plot(piano_buffer)
    axes[2][0].set_xlim([0, len_max])

    S = librosa.feature.melspectrogram(y=original_buffer, sr=fs, n_mels=256)
    L0 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 

#     plots[0][1].set_data(S_dB)
    axes[0][1].cla()
    axes[0][1].imshow(S_dB, aspect='auto', origin='reversed')

    S = librosa.feature.melspectrogram(y=chiptune_buffer, sr=fs, n_mels=256)
    L1 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 

#     plots[1][1].set_data(S_dB)
    axes[1][1].cla()
    axes[1][1].imshow(S_dB, aspect='auto', origin='reversed')

    S = librosa.feature.melspectrogram(y=piano_buffer, sr=fs, n_mels=256)
    L2 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 
    
    axes[2][1].cla()
    axes[2][1].imshow(S_dB, aspect='auto', origin='reversed')

#     plots[2][1].set_data(S_dB)

#     len_max = np.max([len(L0), len(L1), len(L2)])
#     plots[0][2][0].set_ydata(L0)
#     axes[0][2].set_xlim([0, len_max])
#     plots[1][2][0].set_ydata(L1) 
#     axes[1][2].set_xlim([0, len_max])
#     plots[2][2][0].set_ydata(L2)
#     axes[2][2].set_xlim([0, len_max])
    
    len_max = np.max([len(L0), len(L1), len(L2)])
    axes[0][2].cla()
    plots[0][2] = axes[0][2].plot(L0)
    axes[0][2].set_xlim([0, len_max])
    axes[1][2].cla()
    plots[1][2] = axes[1][2].plot(L1) 
    axes[1][2].set_xlim([0, len_max])
    axes[2][2].cla()
    plots[2][2] = axes[2][2].plot(L2)
    axes[2][2].set_xlim([0, len_max])
    
    plt.draw()

if __name__ == "__main__":
    
    path_ori = os.path.join('../data/original/', '*.wav')
    path_bit = os.path.join('../data/8-bits/', '*.wav')
    path_pno = os.path.join('../data/piano/', '*.wav')
    
    files_ori = glob.glob(path_ori)
    files_bit = glob.glob(path_bit)
    files_pno = glob.glob(path_pno)

    files_ori.sort()
    files_bit.sort()
    files_pno.sort()
    
    file_sets = list(zip(files_ori, files_bit, files_pno))
    
    song_names = [file_name.split('/')[-1] for file_name in files_ori]
    
    [print('[' + name + ']') for name in song_names]
    
    song_number = int(input("[Type in your song number]: "))
    song_idx = song_number - 1
    
    keyboard.on_press(keyboard_event_func)
    
    try:
        print('[Target: {}]'.format(song_names[song_idx]))
    except:
        print('[Invalid Song Number: {}]'.format(song_number))
        sys.exit()
    
    original_path, chiptune_path, piano_path = file_sets[song_idx]
    
    print('[fs: {}] [fs: {}] [fs: {}]'.format(librosa.core.get_samplerate(original_path), librosa.core.get_samplerate(chiptune_path), librosa.core.get_samplerate(piano_path)))
    
    fs = 44100
    
    original_stream = librosa.stream(original_path, block_length=1, frame_length=int(fs / 10), hop_length=int(fs/10))
    chiptune_stream = librosa.stream(chiptune_path, block_length=1, frame_length=int(fs / 10), hop_length=int(fs/10))
    piano_stream = librosa.stream(piano_path, block_length=1, frame_length=int(fs / 10), hop_length=int(fs/10))

    original_buffer = np.empty((0), np.float32)
    chiptune_buffer = np.empty((0), np.float32)
    piano_buffer = np.empty((0), np.float32)

    for i, (block_origin, block_chiptune, block_piano) in enumerate(zip(original_stream, chiptune_stream, piano_stream)):

        original_buffer = np.append(original_buffer, block_origin)
        chiptune_buffer = np.append(chiptune_buffer, block_chiptune)
        piano_buffer = np.append(piano_buffer, block_piano)

        if i > 30:
            break

    fig, axes = plt.subplots(3, 3, figsize=(21, 6))

    len_max = np.max([len(original_buffer), len(chiptune_buffer), len(piano_buffer)])
    
    plots = [[None] * 3 for i in range(3)]
    
    plots[0][0] = axes[0][0].plot(original_buffer)
    axes[0][0].set_ylim([-1, 1])
    axes[0][0].set_xlim([0, len_max])

    plots[1][0] =axes[1][0].plot(chiptune_buffer)
    axes[1][0].set_ylim([-1, 1])
    axes[1][0].set_xlim([0, len_max])

    plots[2][0] = axes[2][0].plot(piano_buffer)
    axes[2][0].set_ylim([-1, 1])
    axes[2][0].set_xlim([0, len_max])

    f, t, Zxx = signal.stft(original_buffer, fs, nperseg=int(fs/10))
    specgram = 20 * np.log10(np.maximum(abs(Zxx), 1e-8))
    norm_specgram = (specgram + 160) / 160

    S = librosa.feature.melspectrogram(y=original_buffer, sr=fs, n_mels=256)
    L0 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 

#     axes[0][1].imshow(norm_specgram, aspect='auto', origin='reversed')
    plots[0][1] = axes[0][1].imshow(S_dB, aspect='auto', origin='reversed')

    f, t, Zxx = signal.stft(chiptune_buffer, fs, nperseg=int(fs/10))
    specgram = 20 * np.log10(np.maximum(abs(Zxx), 1e-8))
    norm_specgram = (specgram + 160) / 160

    S = librosa.feature.melspectrogram(y=chiptune_buffer, sr=fs, n_mels=256)
    L1 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 

#     axes[1][1].imshow(norm_specgram, aspect='auto', origin='reversed')
    plots[1][1] = axes[1][1].imshow(S_dB, aspect='auto', origin='reversed')

    f, t, Zxx = signal.stft(piano_buffer, fs, nperseg=int(fs/10))
    specgram = 20 * np.log10(np.maximum(abs(Zxx), 1e-8))
    norm_specgram = (specgram + 160) / 160

    S = librosa.feature.melspectrogram(y=piano_buffer, sr=fs, n_mels=256)
    L2 = np.mean(S, axis=0)
    S_dB = librosa.power_to_db(S, ref=np.max) 

#     axes[2][1].imshow(norm_specgram, aspect='auto', origin='reversed')
    plots[2][1] = axes[2][1].imshow(S_dB, aspect='auto', origin='reversed')

    len_max = np.max([len(L0), len(L1), len(L2)])
    plots[0][2] = axes[0][2].plot(L0)
    axes[0][2].set_xlim([0, len_max])
    plots[1][2] = axes[1][2].plot(L1) 
    axes[1][2].set_xlim([0, len_max])
    plots[2][2] = axes[2][2].plot(L2)
    axes[2][2].set_xlim([0, len_max])
    
    for axes_row in axes:
        for axe in axes_row:
            axe.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    
    plt.tight_layout()
    plt.draw()
    plt.show()