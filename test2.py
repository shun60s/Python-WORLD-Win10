#coding: utf-8

#---------------------------------------------------------------------------------------------
#   Description: a remake of prosody.py, that is read a wav file, encode it and decode from it
#                with process pitch scaling, duration scaling or spectrum warp
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
    parser.add_argument('--scale_warp', '-s', default='pitch_scaling', help='choice one of pitch_scaling,duration_scaling, spectrum_warp ')
    parser.add_argument('--factor', '-f', default=1.5, type=float, help='scale/warp factor')
    parser.add_argument('--use_saved_npy', action='store_true', help='use saved npy dat')
    args = parser.parse_args()

    # load wav file
    wav_path = Path( args.inFILE)
    print('input wave path ',wav_path)
    fs, x_int16 = wavread(wav_path)
    x = x_int16 / (2 ** 15 - 1)
    print ('fs', fs)

    # create WORLD Class object
    vocoder = main.World()

    if args.use_saved_npy:
        print ( 'load dat from saved npy ' + args.inFILE + '.npy' )
        dat=np.load( args.inFILE + '.npy').item()
    else:
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
    	
        # analysis
        # f0_method is 'harvest' and use Requiem
        print ('call vocoder(= main.World).encode ')
        dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True)
        #
        print ( 'save dat as ' + args.inFILE + '.npy' )
        np.save( args.inFILE, dat)
    
    
    
    # process pitch scaling, duration scaling or spectrum warp
    print ( args.scale_warp)
    print ( 'factor=', args.factor)
    # pitch scaling
    if args.scale_warp == 'pitch_scaling':
        # global pitch scaling
        dat = vocoder.scale_pitch(dat, args.factor)
    # duration scaling
    elif args.scale_warp == 'duration_scaling':
    # temporal_positions scaling
        # global duration scaling
        dat = vocoder.scale_duration(dat, args.factor)
    # spectrum warp
    # interp spectrogram
    elif args.scale_warp == 'spectrum_warp':
        dat = vocoder.warp_spectrum(dat, args.factor)
    
    
    if 0:  # fine-grained duration modification
        vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])  # TODO: look into this
    if 0: # show F0 list dump
        print ('*** F0 list ***', len(dat['f0']) )
        print ( dat['f0'])
    
    
    
    # synthesis
    print ('call vocoder(= main.World).decode ')
    dat = vocoder.decode(dat)
    
    # save outputas  wav
    suffix_text= '-' + args.scale_warp + '-' + str(args.factor)
    outFILE=wav_path.stem + suffix_text + '-resynth.wav'
    wavwrite(outFILE, fs, (dat['out'] * 2 ** 15).astype(np.int16))
    print('output wave path ', outFILE)
    
    if 0:  # play audio
        import simpleaudio as sa
        snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
        snd.wait_done()
    if 1:  # visualize
        vocoder.draw(x, dat)


