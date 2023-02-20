"""
Train a Conv-VAD model.

$ python model/train.py --data_path data --epochs 25
"""
import os, sys, argparse
import glob
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, ReLU, LeakyReLU, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN

import tensorflow.keras.backend as K
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.utils import optimize_tf_gpu

optimize_tf_gpu(tf, K)


SHAPE = (400, 126, 1)


def CNN_model():
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
    #model.compile(Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])

    return model


def Small_CNN_model():
    inp = Input(shape=SHAPE)

    x = Conv2D(filters=16,
               kernel_size=3,
               strides=1,
               padding='same',
               use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32,
               kernel_size=3,
               strides=1,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64,
               kernel_size=3,
               strides=2,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    x = Conv2D(filters=128,
               kernel_size=3,
               activation='relu',
               strides=1,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, use_bias=True)(x)
    x = ReLU(6.)(x)

    x = Dense(1, activation='sigmoid')(x)
    out = x

    # create model
    model = Model(inputs=inp, outputs=out)

    return model


def get_model(model_type, weights_path=None):
    """
    Create the VAD model architecture.
    """
    if model_type == 'cnn':
        model = CNN_model()
    elif model_type == 'small_cnn':
        model = Small_CNN_model()
    else:
        raise ValueError('Unsupported model type')

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

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


def train(args):
    log_dir = os.path.join('logs', '000')

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-accuracy{accuracy:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}.h5'),
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, mode='max', patience=20, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='max')
    terminate_on_nan = TerminateOnNaN()
    #checkpoint_clean = CheckpointCleanCallBack(log_dir, max_keep=5)

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]


    # get train&val dataset
    dataset = glob.glob(os.path.join(args.data_path, '*.npy'))
    np.random.shuffle(dataset)
    if args.val_data_path:
        val_dataset = glob.glob(os.path.join(args.data_path, '*.npy'))
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset)*val_split)
        num_train = len(dataset) - num_val

    # get train model
    model = get_model(args.model_type, args.weights_path)
    model.summary()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, args.batch_size))
    model.fit_generator(vad_data_generator(dataset[:num_train], args.batch_size),
            steps_per_epoch=max(1, num_train//args.batch_size),
            validation_data=vad_data_generator(dataset[num_train:], args.batch_size),
            validation_steps=max(1, num_val//args.batch_size),
            epochs=args.epochs,
            initial_epoch=0,
            #verbose=1,
            workers=1,
            use_multiprocessing=False,
            max_queue_size=10,
            callbacks=callbacks)



def main():
    parser = argparse.ArgumentParser(description='train Conv-VAD model')
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='cnn',
        help='Conv-VAD model type: cnn/small_cnn, default=%(default)s')
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--data_path', help='Where training examples are saved.', type=str, required=True)
    parser.add_argument('--val_data_path', help='Where val examples are saved, default=%(default)s', type=str, required=False, default=None)
    parser.add_argument('--val_split', type=float, required=False, default=0.2,
        help = "validation data persentage in dataset if no val dataset provide, default=%(default)s")

    # Training options
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--epochs', type=int, default=20, required=False)

    args = parser.parse_args()

    train(args)



if __name__ == '__main__':
    main()
