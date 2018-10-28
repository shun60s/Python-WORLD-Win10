#coding: utf-8

#---------------------------------------------------------------------------------------------
#   Description: a remake of prosody.py, that is read a wav file, encode it and decode from it
#   Change:      import main as local
#                add argparse
#                visualized ON 1
#   Date:        2018.10
#---------------------------------------------------------------------------------------------
#   This is based on
#         prosody.py
#         of Python-WORLD <https://github.com/tuanad121/Python-WORLD>
#         Copyright (c) 2017
#         Tuan Dinh Anh <dintu@ohsu.edu>
#         Alexander Kain <kaina@ohsu.edu>
#         Pls see LICENSE_Python-WORLD-master.txt in the docs folder
#
#    This WORLD, a high-quality speech analysis/synthesis system on the basis of Vocoder,
#    origins of <http://www.kki.yamanashi.ac.jp/~mmorise/world/> work.
#    
#----------------------------------------------------------------------------------------------

import argparse
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal

# from world import main
import main



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python WORLD')
    parser.add_argument('--inFILE', '-i', default='test-mwm.wav', help='input wav file')
    parser.add_argument('--methodF0', '-m', default='harvest', help='F0 estimation method, harvest or dio ')
    parser.add_argument('--not_requiem', action='store_false', help='use new waveform generator method from WORLD version 0.2.2')
    args = parser.parse_args()

    # load wav file
    wav_path = Path( args.inFILE)
    print('input wave path ',wav_path)
    fs, x_int16 = wavread(wav_path)
    x = x_int16 / (2 ** 15 - 1)
    print ('fs', fs)

    if 0:  # resample
        fs_new = 16000
        x = signal.resample_poly(x, fs_new, fs)
        fs = fs_new

    if 0:  # low-cut
        B = signal.firwin(127, [0.01], pass_zero=False)
        A = np.array([1.0])
        if 0:
            import matplotlib.pyplot as plt
            w, H = signal.freqz(B, A)
            
            fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6))
            ax1.plot(w / np.pi, abs(H))
            ax1.set_ylabel('magnitude')
            ax2.plot(w / np.pi, np.unwrap(np.angle(H)))
            ax2.set_ylabel('unwrapped phase')
            plt.show()
        x = signal.lfilter(B, A, x)

    # create WORLD Class object
    vocoder = main.World()

    # analysis
    print ('call vocoder(= main.World).encode ')
    dat = vocoder.encode(fs, x, f0_method=args.methodF0, is_requiem=args.not_requiem)
    if 0:  # global pitch scaling
        dat = vocoder.scale_pitch(dat, 1.5)
    if 0:  # global duration scaling
        dat = vocoder.scale_duration(dat, 2)
    if 0:  # fine-grained duration modification
        vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])  # TODO: look into this

    if 1: # show F0 list dump
        print ('*** F0 list ***', len(dat['f0']) )
        print ( dat['f0'])

    # synthesis
    print ('call vocoder(= main.World).decode ')
    dat = vocoder.decode(dat)
    
    # save outputas  wav
    suffix_text= '-' + args.methodF0
    if args.not_requiem:
        suffix_text = suffix_text + '-requiem'
    outFILE=wav_path.stem + suffix_text + '-resynth.wav'
    wavwrite(outFILE, fs, (dat['out'] * 2 ** 15).astype(np.int16))
    print('output wave path ', outFILE)
    
    if 0:  # play audio
        import simpleaudio as sa
        snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
        snd.wait_done()
    if 1:  # visualize
        vocoder.draw(x, dat)

