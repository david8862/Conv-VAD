import os, sys
from scipy.io import wavfile
import numpy as np
import conv_vad

import tensorflow.keras.backend as K
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.utils import optimize_tf_gpu

optimize_tf_gpu(tf, K)


# Conv VAD currently only supports single channel audio at a 16k sample rate.
RATE = 16000

# Create a VAD object and load model
vad = conv_vad.VAD()

# Load wav as numpy array
audio = wavfile.read('test.wav')[1].astype(np.uint16)

for i in range(0, audio.shape[0] - RATE, RATE):

    audio_frame = audio[i:i+RATE]

    # For each audio frame (1 sec) compute the speech score.
    # 1 = voice, 0 = no voice
    score = vad.score_speech(audio_frame)
    print('Time =', i // RATE)
    print('Speech Score: ', score)
