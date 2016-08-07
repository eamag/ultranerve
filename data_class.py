from __future__ import print_function

import numpy as np
import cv2
from PIL import Image
import scipy.ndimage as ndi
import os


def number_of_colors(file):
    img = Image.open(file)
    colors = img.getcolors(256)
    return len(colors)


data_path = 'raw/'

image_rows = 420
image_cols = 580


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def random_rotation(x, y, rg=5, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_change(x, y):
    rand = np.random.random()
    if rand < 0.25:
        return random_rotation(x, y)
    elif 0.25 <= rand < 0.5:
        return flip_axis(x, 0), flip_axis(y, 0)
    elif 0.5 <= rand < 0.75:
        return flip_axis(x, 1), flip_axis(y, 1)
    elif 0.75 <= rand < 1:
        return x, y


def preprocess(imgs, img_rows=112, img_cols=144):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    train_len = len(images) / 2
    total = train_len * 3
    not_empty = 2323*3  # count that for you, imgs that have masks
    # generator = ImageDataGenerator(rotation_range=5, height_shift_range=0.01, width_shift_range=0.01,
    #                                horizontal_flip=True, vertical_flip=True, zoom_range=0.1)

    imgs = np.ndarray((not_empty, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_class = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_class = np.ndarray((total, 2), dtype=np.uint8)
    imgs_mask = np.ndarray((not_empty, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    k = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img_class = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        if number_of_colors(os.path.join(train_data_path, image_mask_name)) == 1:  # 2 classes - have mask or not
            img_mask_class = [0, 1]
        else:
            img_mask_class = [1, 0]

            img = np.array([img])
            img_mask = np.array([img_mask])

            imgs[k] = img
            imgs_mask[k] = img_mask
            img, img_mask = random_change(img, img_mask)
            imgs[k + 2323] = img
            imgs_mask[k + 2323] = img_mask
            img, img_mask = random_change(img, img_mask)
            imgs[k + 2323*2] = img
            imgs_mask[k + 2323*2] = img_mask
            k += 1

        img_class = np.array([img_class])
        img_mask_class = np.array([img_mask_class])

        imgs_class[i] = img_class
        imgs_mask_class[i] = img_mask_class
        img_class2, img_class3 = random_change(img_class, img_class)
        imgs_class[i + train_len] = img_class2
        img_class2, img_class3 = random_change(img_class, img_class)
        imgs_class[i + train_len * 2] = img_class3
        imgs_mask_class[i+train_len] = img_mask_class
        imgs_mask_class[i+train_len*2] = img_mask_class

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    imgs = preprocess(imgs)
    imgs_mask = preprocess(imgs_mask)
    imgs_class = preprocess(imgs_class)
    imgs_mask_class = preprocess(imgs_mask_class)
    np.save('imgs_train_not_empty.npy', imgs)
    np.save('imgs_mask_train_not_empty.npy', imgs_mask)
    np.save('imgs_train_class.npy', imgs_class)
    np.save('imgs_mask_train_class.npy', imgs_mask_class)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train_not_empty.npy')
    imgs_mask_train = np.load('imgs_mask_train_not_empty.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_train_data()
    create_test_data()
