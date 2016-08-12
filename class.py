from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, advanced_activations
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from data_class import load_test_data

img_rows = 112
img_cols = 144

smooth = 1.


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    act = advanced_activations.ELU()
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same', init='he_normal')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same', init='he_normal')(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = Convolution2D(32, 3, 3, activation=act, border_mode='same', init='he_normal', subsample=(2, 2))(norm1)

    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same', init='he_normal')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same', init='he_normal')(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = Convolution2D(64, 3, 3, activation=act, border_mode='same', init='he_normal', subsample=(2, 2))(norm2)

    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same', init='he_normal')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same', init='he_normal')(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = Convolution2D(128, 3, 3, activation=act, border_mode='same', init='he_normal', subsample=(2, 2))(norm3)

    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same', init='he_normal')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same', init='he_normal')(conv4)
    norm4 = BatchNormalization()(conv4)
    pool4 = Convolution2D(256, 3, 3, activation=act, border_mode='same', init='he_normal', subsample=(2, 2))(norm4)

    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same', init='he_normal')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same', init='he_normal')(conv5)
    norm5 = BatchNormalization()(conv5)
    #
    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(up6)
    # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv6)
    #
    # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(up7)
    # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv7)
    #
    # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(up8)
    # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv8)
    #
    # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(up9)
    # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv9)

    flatten = Flatten()(norm5)  # 288
    dense2 = Dense(16, activation=act, init='he_normal')(flatten)
    dense = Dense(2, activation='sigmoid', init='he_normal')(dense2)

    model = Model(input=inputs, output=dense)

    model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load('imgs_train_class.npy')
    imgs_mask_train = np.load('imgs_mask_train_class.npy')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet_class.hdf5', monitor='loss', save_best_only=True, save_weights_only=True)

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('unet_class.hdf5')

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.01, height_shift_range=0.01, zoom_range=0.1,
                                 horizontal_flip=True, vertical_flip=True)
    model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=24, shuffle=True),
                        samples_per_epoch=len(imgs_train) * 2, nb_epoch=20, verbose=1, callbacks=[model_checkpoint])

    # model.fit(imgs_train, imgs_mask_train, batch_size=24, nb_epoch=1, verbose=1, shuffle=True,
    #           callbacks=[model_checkpoint], validation_split=0.5)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading model...')
    print('-' * 30)
    model.load_weights('unet_class.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test_class.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
