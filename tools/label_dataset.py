#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert & label clip wav audios to melspec feature numpy array files
"""
import os, sys, argparse
import glob
import wave
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import uuid

import librosa
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db


def show_audio_info(wf):
    print('channels: {}'.format(wf.getnchannels()))
    print('sample rate: {}'.format(wf.getframerate()))
    print('bits per sample: {}'.format(wf.getsampwidth() * 8))
    print('total frames: {}'.format(wf.getnframes()))
    print('duration seconds: {} s'.format(wf.getnframes() / wf.getframerate()))
    print('compress type: {}'.format(wf.getcomptype()))
    print('compress name: {}'.format(wf.getcompname()))


def wav_check(wav_file, channel_num, sample_rate, sample_bit):
    wf = wave.open(wav_file, 'rb')

    if wf.getnchannels() != channel_num or wf.getframerate() != sample_rate or wf.getsampwidth() * 8 != sample_bit:
        # break if any invalid audio file
        print('Invalid wav audio file:', wav_file)
        print('\nAudio info:')
        show_audio_info(wf)
        exit()

    wav_length = (wf.getnframes() / wf.getframerate())
    wf.close()

    return wav_length



RATE = 16000
def convert_wav_file(wav_file, output_path, label):
    wav_data = wavfile.read(wav_file)[1].astype(np.int16)
    wav_data = wav_data[:RATE]

    if len(wav_data) < RATE:
        wav_data = np.concatenate([np.zeros((RATE - len(wav_data),)), wav_data])

    melspec = do_melspec(y=wav_data.astype(np.float32), sr=RATE, n_mels=416, fmax=4000, hop_length=128)
    norm_melspec = pwr_to_db(melspec, ref=np.max)
    spectrogram = (1 - (norm_melspec / -80.0))[:-16, :]

    fn = os.path.join(output_path, '{}_{}.npy'.format(label, uuid.uuid4().hex))
    np.save(fn, spectrogram)



def main():
    parser = argparse.ArgumentParser(description='convert & label clip wav audios to melspec feature numpy array files')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to label')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save numpy array file')
    parser.add_argument('--label', type=str, required=True, choices=['voice', 'noise'],
                        help='wav audio label type')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        wav_check(args.wav_path, channel_num=1, sample_rate=RATE, sample_bit=16)
        convert_wav_file(args.wav_path, args.output_path, args.label)

    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))

        pbar = tqdm(total=len(wav_files), desc='label wav audio')
        for wav_file in wav_files:
            wav_check(wav_file, channel_num=1, sample_rate=RATE, sample_bit=16)
            convert_wav_file(wav_file, args.output_path, args.label)
            pbar.update(1)
        pbar.close()
    print('\nDone')


if __name__ == "__main__":
    main()
