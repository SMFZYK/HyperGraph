# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.20
import csv

import scipy.io as scio
import numpy as np
import torch


def load_ft(data_dir, feature_name='GVCNN'):
    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test


def read_csv(data_dir):
    with open(data_dir, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        # md_data_new = ToTensor(md_data)
        return torch.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def read_txt(data_dir):
    with open(data_dir, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return torch.FloatTensor(md_data)


def load_ft_MDA(data_dir, feature_name='GVCNN'):
    data = read_csv(data_dir)
    if feature_name == 'MVCNN':
        # fts = data.item().astype(np.float32)
        fts = data.numpy()
        fts = fts.astype(np.float32)
    elif feature_name == 'GVCNN':
        # fts = data.astype(np.float32)
        fts = data.numpy()
        fts = fts.astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    return fts
