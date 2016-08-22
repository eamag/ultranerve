# convert predicted class data for u-net

from __future__ import print_function

import numpy as np
import cv2
from data_class import image_cols, image_rows


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def pre_train():
    from data_class import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test_class = np.load('imgs_mask_test_class.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    imgs_mask_test_class = imgs_mask_test_class[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        if imgs_mask_test_class[i][0] > imgs_mask_test_class[i][1]:
            rle = img
            ids.append(imgs_id_test[i])
            rles.append(rle)
        else:
            continue

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    imgs_mask = np.ndarray((len(rles), 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((len(rles),), dtype=np.int32)
    for i in range(len(rles)):
        imgs_mask[i, 0] = rles[i]
        imgs_id[i] = ids[i]
    np.save('imgs_test_not_empty.npy', imgs_mask)
    np.save('imgs_id_test_not_empty.npy', imgs_id)
    print(len(imgs_mask))

if __name__ == '__main__':
    pre_train()
