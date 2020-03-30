#-*-coding:utf-8-*-
from __future__ import print_function
import os
import gzip
import sys
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs

def gzip_mnist(path=""):
    root = sys.path[0]
    raw_folder = "fashionmnist_data/FashionMNIST/raw" 
    filenames = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file in filenames:
        file_path = os.path.join(root, raw_folder, file)
        with open(file_path.replace('.gz', ''), 'wb') as ouf_f, gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert(get_int(data[:4])) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert(get_int(data[:4])) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

def _training_file(self, split):
    return 'training_{}.pt'.format(split)

def _test_file(self, split):
    return 'test_{}.pt'.format(split)

def pre_process():
    root = sys.path[0]
    raw_folder = "fashionmnist_data/FashionMNIST/raw"

    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    raw_folder = os.path.join(root, raw_folder)
    gzip_folder = os.path.join(raw_folder, 'gzip')
    processed_folder= "fashionmnist_data/FashionMNIST/processed"
    training_file = 'training.pt'
    test_file = 'test.pt'
    training_set = (read_image_file(os.path.join(root, raw_folder, 'train-images-idx3-ubyte')),read_label_file(os.path.join(root, raw_folder, 'train-labels-idx1-ubyte')))
    test_set = (read_image_file(os.path.join(root, raw_folder, 't10k-images-idx3-ubyte')),read_label_file(os.path.join(root, raw_folder, 't10k-labels-idx1-ubyte')))
    with open(os.path.join(root, processed_folder, training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(root, processed_folder, test_file), 'wb') as f:
        torch.save(test_set, f)
    print('Done!')

if __name__ == "__main__":
    if (int(input()) == 1):
        gzip_mnist()
    pre_process()
