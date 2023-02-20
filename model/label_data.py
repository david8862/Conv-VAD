"""
Create training data interactively with a .wav file.

$ python model/label_data.py --wav_path path/to/audio.wav --data_path data
"""
from datetime import datetime
from scipy.io import wavfile
import pyaudio
import numpy as np
import click
import os

import librosa
do_melspec = librosa.feature.melspectrogram
pwr_to_db = librosa.core.power_to_db

# Use older model to optimize labeling
try:
    import conv_vad
    vad = conv_vad.VAD()
except ImportError:
    vad = None


RATE = 16000


def save_example_feature(dataset_path, audio_frame, label):

    melspec = do_melspec(y=audio_frame.astype(np.float32), sr=RATE, n_mels=416, fmax=4000, hop_length=128)
    norm_melspec = pwr_to_db(melspec, ref=np.max)
    spectrogram = (1 - (norm_melspec / -80.0))[:-16, :]
    fn = os.path.join(dataset_path, '{}_{}.npy'.format(label, datetime.now().strftime('%Y%m%d%M%S')))
    np.save(fn, spectrogram)


def save_example(dataset_path, audio_frame, label):
    import wavio
    fn = os.path.join(dataset_path, '{}_{}.wav'.format(label, datetime.now().strftime('%Y%m%d%M%S')))
    wavio.write(fn, audio_frame, RATE, sampwidth=2, scale='none')


@click.command()
@click.option('--wav_path',
              required=True,
              help='The .wav file (rate=16000, channels=1) used create training examples.',
              type=click.Path())
@click.option('--data_path',
              required=True,
              help='Where to save training examples.',
              type=click.Path())
@click.option('--assist/--no-assist',
              required=False,
              help='Use previous model to assist labeling.',
              default=False)
@click.option('--random/--no-random',
              required=False,
              help='Label segments randomly.',
              default=True)
def make_labels(wav_path=None, data_path=None, assist=None, random=None):

    os.makedirs(data_path, exist_ok=True)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    frames_per_buffer=RATE,
                    output=True)

    wav_data = wavfile.read(wav_path)[1].astype(np.int16)
    data_length = wav_data.shape[0]

    if not random:
        counter = {'pos': 0}

    def next_idx():
        if random:
            return np.random.randint(0, data_length - RATE)
        else:
            prev = counter['pos']
            counter['pos'] += RATE
            return prev

    idx = next_idx()

    while True:

        audio_frame = wav_data[idx:idx+RATE]

        if vad is not None:
            score = vad.score_speech(audio_frame)
            print('score =', score)

            # Skip confident classifications
            if assist:
                if score < 0.25 or score > 0.6:
                    idx = next_idx()
                    continue

        stream.write(audio_frame.tobytes())

        opt = input('quit (q) / skip (s) / voice (v) / noise (n) > ').lower()
        if 'q' in opt:
            stream.close()
            p.terminate()
            break
        elif 's' in opt:
            idx = next_idx()
        elif 'v' in opt:
            save_example(data_path, audio_frame, 'voice')
            idx = next_idx()
        elif 'n' in opt:
            save_example(data_path, audio_frame, 'noise')
            idx = next_idx()


if __name__ == '__main__':
    make_labels()
