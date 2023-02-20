"""
Train a Conv-VAD model.

$ python model/train.py --data_path data --epochs 25
"""
import os, sys, argparse
import glob
import numpy as np


from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *


SHAPE = (400, 126, 1)


def get_model():
    """
    Create the VAD model architecture.
    """
    inp = Input(shape=SHAPE)

    x = Conv2D(64, (9, 9))(inp)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (5, 5))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(256, (3, 3))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = LeakyReLU()(x)

    x = Dense(1, activation='sigmoid')(x)
    out = x

    model = Model(inputs=inp, outputs=out)
    model.compile(Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])

    return model


def vad_data_generator(sample_list, batch_size):
    sample_num = len(sample_list)
    i = 0

    while True:
        X, Y = [], []

        for b in range(batch_size):
            if i==0:
                np.random.shuffle(sample_list)

            sample = sample_list[i]
            feature_array = np.load(sample).reshape(*SHAPE)
            X.append(feature_array)
            Y.append(1 if 'voice' in sample else 0)
            i = (i+1) % sample_num

        X = np.array(X)
        Y = np.array(Y)
        # Precomputed Normalization
        X = (X - 0.643) / 0.094

        yield X, Y



def train_bk(data_path=None, model_path=None, epochs=None, batch_size=None):

    X, Y = [], []

    print('Loading data...', end='')
    for fn in glob.iglob(os.path.join(data_path, '*.npy')):
        ary = np.load(fn).reshape(*SHAPE)
        X.append(ary)
        Y.append(1 if 'voice' in fn else 0)
    X = np.array(X)
    Y = np.array(Y)

    shuffle_idxs = np.random.permutation(X.shape[0])
    X = X[shuffle_idxs]
    Y = Y[shuffle_idxs]
    print('done.')

    print('X stats ->', X.mean(), X.std(), X.shape)
    print('Y stats ->', Y.mean(), Y.std(), Y.shape)

    # Precomputed Normalization
    X = (X - 0.643) / 0.094

    model = get_model()
    model.summary()
    model.fit(X, Y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              callbacks=[ModelCheckpoint(filepath=model_path, save_best_only=True)])


def train(data_path, model_path, epochs, batch_size):
    sample_list = glob.glob(os.path.join(data_path, '*.npy'))
    sample_num = len(sample_list)

    model = get_model()
    model.summary()

    model.fit_generator(vad_data_generator(sample_list, batch_size),
            steps_per_epoch=max(1, sample_num//batch_size),
            #validation_data=val_data_generator,
            #validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape, anchors, num_classes, multi_anchor_assign=args.multi_anchor_assign),
            #validation_steps=max(1, num_val//args.batch_size),
            epochs=epochs,
            #initial_epoch=0,
            #verbose=1,
            #batch_size=batch_size,
            #validation_split=0.2,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=[ModelCheckpoint(filepath=model_path, save_best_only=True)])



def main():
    parser = argparse.ArgumentParser(description='validate YOLO model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--data_path', help='Where training examples are saved.', type=str, required=True)
    parser.add_argument('--model_path', help='Where to save trained model.', type=str, default='vad_best.h5', required=True)
    parser.add_argument('--epochs', type=int, default=20, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)

    args = parser.parse_args()

    train(args.data_path, args.model_path, args.epochs, args.batch_size)



if __name__ == '__main__':
    main()
