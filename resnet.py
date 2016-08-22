from __future__ import print_function
from __future__ import absolute_import

# import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, advanced_activations
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# from data_class import load_test_data

img_rows = 224
img_cols = 224

smooth = 1.

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
# 50 Layer resnet
def resnet():
    input = Input((1, 224, 224))

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
    block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
    block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)

    # Classifier block
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
    flatten1 = Flatten()(pool2)
    dense = Dense(output_dim=2, init="he_normal", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

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
    imgs_train = np.load('imgs_train_resnet.npy')
    imgs_mask_train = np.load('imgs_mask_train_resnet.npy')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = resnet()
    model_checkpoint = ModelCheckpoint('resnet.hdf5', monitor='loss', save_best_only=True)

    # print('-' * 30)
    # print('Loading saved weights...') , save_weights_only=True
    # print('-' * 30)
    # model.load_weights('unet_class.hdf5')

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    # datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.01, height_shift_range=0.01, zoom_range=0.1,
    #                              horizontal_flip=True, vertical_flip=True)
    # model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=24, shuffle=True),
    #                     samples_per_epoch=len(imgs_train) * 2, nb_epoch=20, verbose=1, callbacks=[model_checkpoint])

    model.fit(imgs_train, imgs_mask_train, batch_size=8, nb_epoch=1, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_split=0.5)

    # print('-' * 30)
    # print('Loading and preprocessing test data...')
    # print('-' * 30)
    # imgs_test, imgs_id_test = load_test_data()
    # imgs_test = preprocess(imgs_test)
    #
    # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std
    #
    # print('-' * 30)
    # print('Loading model...')
    # print('-' * 30)
    # model.load_weights('resnet.hdf5')
    #
    # print('-' * 30)
    # print('Predicting masks on test data...')
    # print('-' * 30)
    # imgs_mask_test = model.predict(imgs_test, verbose=1)
    # np.save('imgs_mask_test_class.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
