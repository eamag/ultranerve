from __future__ import print_function

import numpy as np
import cv2
from data_class import image_cols, image_rows


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 5)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    from data_class import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test_not_empty.npy')
    imgs_mask_test_class = np.load('imgs_mask_test_class.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_mask_test_class = imgs_mask_test_class[argsort]

    imgs_id_test_not_empty = np.load('imgs_id_test_not_empty.npy')
    argsort2 = np.argsort(imgs_id_test_not_empty)
    imgs_test = imgs_test[argsort2]

    total = 5508  # imgs in test folder
    ids = []
    rles = []
    k = 0
    for i in range(total):

        if imgs_mask_test_class[i][0] < imgs_mask_test_class[i][1]:
            rle = ''
        else:
            img = imgs_test[k, 0]
            img = prep(img)
            rle = run_length_enc(img)
            k += 1
            if k % 50 == 0:
                print('{}/{}'.format(k, len(imgs_test)))

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 5500 == 0:
            print('{}/{}'.format(i, total))
    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    submission()
