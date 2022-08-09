# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.20

# from datasets import load_ft
# from datasets.data_helper import load_ft_MDA
# from utils import hypergraph_utils as hgut


# def load_feature_construct_H(data_dir,
#                              m_prob=1,
#                              K_neigs=[10],
#                              is_probH=True,
#                              split_diff_scale=False,
#                              use_mvcnn_feature=False,
#                              use_gvcnn_feature=True,
#                              use_mvcnn_feature_for_structure=False,
#                              use_gvcnn_feature_for_structure=True):
#     """
#
#     :param data_dir: directory of feature data
#     :param m_prob: parameter in hypergraph incidence matrix construction 超图关联矩阵构造中的参数
#     :param K_neigs: the number of neighbor expansion 邻居扩展数
#     :param is_probH: probability Vertex-Edge matrix or binary 概率顶点边缘矩阵或二进制
#     :param use_mvcnn_feature:
#     :param use_gvcnn_feature:
#     :param use_mvcnn_feature_for_structure:
#     :param use_gvcnn_feature_for_structure:
#     :return:
#     """
#     # init feature
#     if use_mvcnn_feature or use_mvcnn_feature_for_structure:
#         mvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='MVCNN')
#     if use_gvcnn_feature or use_gvcnn_feature_for_structure:
#         gvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='GVCNN')
#     if 'mvcnn_ft' not in dir() and 'gvcnn_ft' not in dir():
#         raise Exception('None feature initialized')
#
#     # construct feature matrix
#     fts = None
#     if use_mvcnn_feature:
#         fts = hgut.feature_concat(fts, mvcnn_ft)
#     if use_gvcnn_feature:
#         fts = hgut.feature_concat(fts, gvcnn_ft)
#     if fts is None:
#         raise Exception(f'None feature used for model!')
#
#     # construct hypergraph incidence matrix
#     print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
#     H = None
#     if use_mvcnn_feature_for_structure:
#         tmp = hgut.construct_H_with_KNN(mvcnn_ft, K_neigs=K_neigs,
#                                         split_diff_scale=split_diff_scale,
#                                         is_probH=is_probH, m_prob=m_prob)
#         H = hgut.hyperedge_concat(H, tmp)
#     if use_gvcnn_feature_for_structure:
#         tmp = hgut.construct_H_with_KNN(gvcnn_ft, K_neigs=K_neigs,
#                                         split_diff_scale=split_diff_scale,
#                                         is_probH=is_probH, m_prob=m_prob)
#         H = hgut.hyperedge_concat(H, tmp)
#     if H is None:
#         raise Exception('None feature to construct hypergraph incidence matrix!')
#
#     return fts, lbls, idx_train, idx_test, H
from datasets.data_helper import load_ft_MDA
from utils import hypergraph_utils as hgut

def load_feature_construct_H(data_dir,
                             K_neigs,
                             m_prob=0.8,
                             is_probH=True,
                             split_diff_scale=False,
                             use_mvcnn_feature=True,
                             use_gvcnn_feature=True,
                             use_mvcnn_feature_for_structure=True,
                             use_gvcnn_feature_for_structure=True):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction 超图关联矩阵构造中的参数
    :param K_neigs: the number of neighbor expansion 邻居扩展数
    :param is_probH: probability Vertex-Edge matrix or binary 概率顶点边缘矩阵或二进制
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    # init feature
    if use_mvcnn_feature or use_mvcnn_feature_for_structure:
        # mvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='MVCNN')
        mvcnn_ft = load_ft_MDA(data_dir, feature_name='MVCNN')
    if use_gvcnn_feature or use_gvcnn_feature_for_structure:
        # gvcnn_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='GVCNN')
        gvcnn_ft = load_ft_MDA(data_dir, feature_name='GVCNN')
    if 'mvcnn_ft' not in dir() and 'gvcnn_ft' not in dir():
        raise Exception('None feature initialized')

    # construct feature matrix
    fts = None
    if use_mvcnn_feature:
        fts = hgut.feature_concat(fts, mvcnn_ft)
    if use_gvcnn_feature:
        fts = hgut.feature_concat(fts, gvcnn_ft)
    if fts is None:
        raise Exception(f'None feature used for model!')

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    if use_mvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(mvcnn_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp)
    if use_gvcnn_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(gvcnn_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return fts, H
