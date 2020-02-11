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
from matplotlib.widgets import SpanSelector

ori_indmin = 0
ori_indmax = -1

chip_indmin = 0
chip_indmax = -1

pia_indmin = 0
pia_indmax = -1

segment_idx = 0

def onselect_original(xmin, xmax):
    
    global ori_indmin
    global ori_indmax
    
    print('[MIN: {} / MAX: {}]'.format(xmin, xmax))
#     indmin, indmax = np.searchsorted(original_buffer, (xmin, xmax))
    indmin = int(xmin)
    indmax = int(xmax)
    indmax = min(len(original_buffer) - 1, indmax)

    print('[Selected: {} ~ {}]'.format(indmin, indmax))
    ori_indmin, ori_indmax = indmin, indmax
#     thisx = x[indmin:indmax]
#     thisy = y[indmin:indmax]
#     line2.set_data(thisx, thisy)
#     ax2.set_xlim(thisx[0], thisx[-1])
#     ax2.set_ylim(thisy.min(), thisy.max())
#     fig.canvas.draw()

def onselect_chiptune(xmin, xmax):
    
    global chip_indmin
    global chip_indmax
    
    print('[MIN: {} / MAX: {}]'.format(xmin, xmax))
    indmin = int(xmin)
    indmax = int(xmax)
    indmax = min(len(chiptune_buffer) - 1, indmax)

    print('[Selected: {} ~ {}]'.format(indmin, indmax))
    chip_indmin, chip_indmax = indmin, indmax

    
def onselect_piano(xmin, xmax):
    
    global pia_indmin
    global pia_indmax
    
    print('[MIN: {} / MAX: {}]'.format(xmin, xmax))
    indmin = int(xmin)
    indmax = int(xmax)
    indmax = min(len(piano_buffer) - 1, indmax)

    print('[Selected: {} ~ {}]'.format(indmin, indmax))
    pia_indmin, pia_indmax = indmin, indmax

def keyboard_event_func(e):
    
    global segment_idx
    global save_file_name_prefix
    
    global original_buffer
    global chiptune_buffer
    global piano_buffer
    
    global original_stream
    global chiptune_stream
    global piano_stream
    
    global ori_indmin
    global ori_indmax
    
    global chip_indmin
    global chip_indmax
    
    global pia_indmin
    global pia_indmax
    
#     if e.name == 'delete':
    if e.name == 'p':
        sd.play(original_buffer[ori_indmin:ori_indmax], fs)
        
    elif e.name == 'l':
        sd.play(chiptune_buffer[chip_indmin:chip_indmax], fs)
    
    elif e.name == ',':
        sd.play(piano_buffer[pia_indmin:pia_indmax], fs)
                
    elif e.name == 'k':
        len_max = max(ori_indmax - ori_indmin, chip_indmax - chip_indmin)
        two_chan_audio = np.zeros([len_max, 2])
        two_chan_audio[:ori_indmax - ori_indmin, 0] = original_buffer[ori_indmin:ori_indmax]
        two_chan_audio[:chip_indmax - chip_indmin, 1] = chiptune_buffer[chip_indmin:chip_indmax]
        print(two_chan_audio.shape)
        
        sd.play(two_chan_audio, fs)
    
    elif e.name == 'm':
        len_max = max(ori_indmax - ori_indmin, pia_indmax - pia_indmin)
        two_chan_audio = np.zeros([len_max, 2])
        two_chan_audio[:ori_indmax - ori_indmin, 0] = original_buffer[ori_indmin:ori_indmax]
        two_chan_audio[:pia_indmax - pia_indmin, 1] = piano_buffer[pia_indmin:pia_indmax]
        print(two_chan_audio.shape)
        
        sd.play(two_chan_audio, fs)
        
    # One Step Block Loading
        
    elif e.name == 'q':
        original_buffer = np.append(original_buffer, next(original_stream))
        plot_figure()
        
    elif e.name == 'a':
        chiptune_buffer = np.append(chiptune_buffer, next(chiptune_stream))
        plot_figure()
    
    elif e.name == 'z':
        piano_buffer = np.append(piano_buffer, next(piano_stream))
        plot_figure()
        
    # One Step Block Forwarding
        
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
        
    # Ten Step Block Forwarding
        
    elif e.name == 'e':
        for i in range(10):
            _len = len(original_buffer)
            original_buffer = np.append(original_buffer, next(original_stream))
            original_buffer = original_buffer[-_len:]
        
        plot_figure()
        
    elif e.name == 'd':
        for i in range(10):
            _len = len(chiptune_buffer)
            chiptune_buffer = np.append(chiptune_buffer, next(chiptune_stream))
            chiptune_buffer = chiptune_buffer[-_len:]

        plot_figure()
    
    elif e.name == 'c':
        for i in range(10):
            _len = len(piano_buffer)
            piano_buffer = np.append(piano_buffer, next(piano_stream))
            piano_buffer = piano_buffer[-_len:]
            
        plot_figure()
        
    elif e.name == '\\':
    
        len_max = np.max([ori_indmax - ori_indmin, chip_indmax - chip_indmin, pia_indmax - pia_indmin])
        
        npy_array = np.zeros([len_max, 3])
        npy_array[:ori_indmax - ori_indmin, 0] = original_buffer[ori_indmin:ori_indmax]
        npy_array[:chip_indmax - chip_indmin, 1] = chiptune_buffer[chip_indmin:chip_indmax]
#         npy_array[:pia_indmax - pia_indmin, 2] = piano_buffer[pia_indmin:pia_indmax]1
        
        npy_array = np.zeros([len_max, 3])
        npy_array[:ori_indmax - ori_indmin, 0] = original_buffer[ori_indmin:ori_indmax]
        npy_array[:chip_indmax - chip_indmin, 1] = chiptune_buffer[chip_indmin:chip_indmax]
#         npy_array[:pia_indmax - pia_indmin, 2] = piano_buffer[pia_indmin:pia_indmax]
        
        print(npy_array.shape)
        segment_file_name = save_file_name_prefix + '_{:03d}.npy'.format(segment_idx)
        print(segment_file_name)
        np.save(segment_file_name, npy_array)
        segment_idx += 1
    
        for i in range(int(np.floor(10 * ori_indmax/fs) - 1)):
            _len = len(original_buffer)
            original_buffer = np.append(original_buffer, next(original_stream))
            original_buffer = original_buffer[-_len:]
            
        for i in range(int(np.floor(10 * chip_indmax/fs) - 1)):
            _len = len(chiptune_buffer)
            chiptune_buffer = np.append(chiptune_buffer, next(chiptune_stream))
            chiptune_buffer = chiptune_buffer[-_len:]
    
        for i in range(int(np.floor(10 * pia_indmax/fs) - 1)):
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
    axes[0][0].set_ylim([-1, 1])


#     plots[1][0][0].set_ydata(chiptune_buffer)
    axes[1][0].cla()
    axes[1][0].plot(chiptune_buffer)
    axes[1][0].set_xlim([0, len_max])
    axes[1][0].set_ylim([-1, 1])

#     plots[2][0][0].set_ydata(piano_buffer)
    axes[2][0].cla()
    axes[2][0].plot(piano_buffer)
    axes[2][0].set_xlim([0, len_max])
    axes[2][0].set_ylim([-1, 1])

    S = librosa.feature.melspectrogram(y=original_buffer, sr=fs, n_mels=256)
    L0 = np.mean(S, axis=0)
    S_dB_1 = librosa.power_to_db(S, ref=np.max) 

#     plots[0][1].set_data(S_dB)
    axes[0][1].cla()
#     axes[0][1].imshow(S_dB, aspect='auto', origin='reversed')

    S = librosa.feature.melspectrogram(y=chiptune_buffer, sr=fs, n_mels=256)
    L1 = np.mean(S, axis=0)
    S_dB_2 = librosa.power_to_db(S, ref=np.max) 

#     plots[1][1].set_data(S_dB)
    axes[1][1].cla()
#     axes[1][1].imshow(S_dB, aspect='auto', origin='reversed')

    S = librosa.feature.melspectrogram(y=piano_buffer, sr=fs, n_mels=256)
    L2 = np.mean(S, axis=0)
    S_dB_3 = librosa.power_to_db(S, ref=np.max) 
    
    axes[2][1].cla()
#     axes[2][1].imshow(S_dB, aspect='auto', origin='reversed')
    
    len_max = np.max([S_dB_1.shape[1], S_dB_1.shape[1], S_dB_1.shape[1]])

    plots[0][1] = axes[0][1].imshow(S_dB_1, aspect='auto', origin='reversed')
    axes[0][1].set_xlim([0, len_max])
    plots[1][1] = axes[1][1].imshow(S_dB_2, aspect='auto', origin='reversed')
    axes[1][1].set_xlim([0, len_max])
    plots[2][1] = axes[2][1].imshow(S_dB_3, aspect='auto', origin='reversed')
    axes[2][1].set_xlim([0, len_max])
    

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
        save_file_name_prefix = os.path.join('../segment', song_names[song_idx])
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

    once = True
    
    for i, (block_origin, block_chiptune, block_piano) in enumerate(zip(original_stream, chiptune_stream, piano_stream)):
        
        if once:
            print('[Data Type: {} / {} / {}]'.format(block_origin.dtype, block_chiptune.dtype, block_piano.dtype))
            once = False
        
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
    S_dB_1 = librosa.power_to_db(S, ref=np.max) 

#     axes[0][1].imshow(norm_specgram, aspect='auto', origin='reversed')

    f, t, Zxx = signal.stft(chiptune_buffer, fs, nperseg=int(fs/10))
    specgram = 20 * np.log10(np.maximum(abs(Zxx), 1e-8))
    norm_specgram = (specgram + 160) / 160

    S = librosa.feature.melspectrogram(y=chiptune_buffer, sr=fs, n_mels=256)
    L1 = np.mean(S, axis=0)
    S_dB_2 = librosa.power_to_db(S, ref=np.max) 

#     axes[1][1].imshow(norm_specgram, aspect='auto', origin='reversed')

    f, t, Zxx = signal.stft(piano_buffer, fs, nperseg=int(fs/10))
    specgram = 20 * np.log10(np.maximum(abs(Zxx), 1e-8))
    norm_specgram = (specgram + 160) / 160

    S = librosa.feature.melspectrogram(y=piano_buffer, sr=fs, n_mels=256)
    L2 = np.mean(S, axis=0)
    S_dB_3 = librosa.power_to_db(S, ref=np.max) 

#     axes[2][1].imshow(norm_specgram, aspect='auto', origin='reversed')

    len_max = np.max([S_dB_1.shape[1], S_dB_1.shape[1], S_dB_1.shape[1]])

    plots[0][1] = axes[0][1].imshow(S_dB_1, aspect='auto', origin='reversed')
    axes[0][1].set_xlim([0, len_max])
    plots[1][1] = axes[1][1].imshow(S_dB_2, aspect='auto', origin='reversed')
    axes[1][1].set_xlim([0, len_max])
    plots[2][1] = axes[2][1].imshow(S_dB_3, aspect='auto', origin='reversed')
    axes[2][1].set_xlim([0, len_max])

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
    
    span_original = SpanSelector(axes[0][0], onselect_original, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
    span_chiptune = SpanSelector(axes[1][0], onselect_chiptune, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
    span_piano = SpanSelector(axes[2][0], onselect_piano, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
    
    
    
    plt.show()