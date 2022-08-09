# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.16
import csv

import random

import torch
from torchvision.transforms import ToTensor


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
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


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return torch.FloatTensor(md_data)


def prepare_data(opt):
    dataset = dict()

    # 阅读 miRNA-Disease 关联矩阵
    dataset['md_p'] = read_csv(opt.data_path + '\\m-d.csv')  # Tensor: (495, 383)
    dataset['md_true'] = read_csv(opt.data_path + '\\m-d.csv')

    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)  # (5430, 2)
    random.shuffle(zero_index)  # (184155, 2)
    # 超参数
    length = int(len(one_index) * 0.1)
    cha_index = one_index[0:length]
    cha_index0 = zero_index[0:length]
    cha_index1 = one_index[length:2 * length]
    cha_index2 = zero_index[length:2 * length]
    zero_index.extend(one_index[0:2 * length])
    one_index = one_index[2 * length:len(one_index)]
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)
    for ind in cha_index:
        if dataset['md_p'][ind[0], ind[1]] != 0:
            dataset['md_p'][ind[0], ind[1]] = 0
        else:
            print("md矩阵有错！")

    dataset['md_p_cha'] = dataset['md_true'] - dataset['md_p']
    dataset['md_true'] = dataset['md_p']

    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    # 阅读 Disease-Disease 相似度矩阵
    dd_matrix = read_csv(opt.data_path + '\\d-d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    # 阅读 miRNA-miRNA 相似度矩阵
    mm_matrix = read_csv(opt.data_path + '\\m-m.csv')
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset, cha_index, cha_index0, cha_index1, cha_index2


if __name__ == '__main__':
    # a = read_csv("../dataset/data(383-495)\\m-d.csv")
    pass
