from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, UpSampling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator

from data_class import load_test_data

img_rows = 112
img_cols = 144

smooth = 1.


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv1)
    pool1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal', subsample=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv2)
    pool2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal', subsample=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv3)
    pool3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal', subsample=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv4)
    pool4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal', subsample=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='he_normal')(conv9)

    flatten = Flatten()(conv9)  # 288
    dense2 = Dense(16, activation='relu', init='he_normal')(flatten)
    dense = Dense(2, activation='softmax', init='he_normal')(dense2)

    model = Model(input=inputs, output=dense)

    model.compile(optimizer=Adam(lr=1e-15), loss='binary_crossentropy', metrics=['accuracy'])

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
    model.fit(imgs_train, imgs_mask_train, batch_size=10, nb_epoch=3, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_split=0.2)
    # datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.01, height_shift_range=0.01, zoom_range=0.1,
    #                              horizontal_flip=True, vertical_flip=True)
    # model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=20, shuffle=True),
    #                     samples_per_epoch=len(imgs_train) * 2, nb_epoch=100, verbose=1, callbacks=[model_checkpoint])

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
