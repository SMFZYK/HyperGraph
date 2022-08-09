# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.16
import torch
from torch import nn
from torch_geometric.nn import conv
import torch.nn.functional as F

# class Model(nn.Module):
#     def __init__(self, sizes):
#         super(Model, self).__init__()
#
#         self.fg = sizes.fg
#         self.fd = sizes.fd
#         self.k = sizes.k
#         self.m = sizes.m
#         self.d = sizes.d
#         self.gcn_x1 = conv.GCNConv(self.fg, self.fg)
#         self.gcn_y1 = conv.GCNConv(self.fd, self.fd)
#         self.gcn_x2 = conv.GCNConv(self.fg, self.fg)
#         self.gcn_y2 = conv.GCNConv(self.fd, self.fd)
#
#         # self.linear_gcn_x1 = nn.Linear(self.fg, self.fg)
#         # self.linear_gcn_y1 = nn.Linear(self.fd, self.fd)
#         # self.linear_gcn_x2 = nn.Linear(self.fg, self.fg)
#         # self.linear_gcn_y2 = nn.Linear(self.fd, self.fd)
#
#         self.linear_x_1 = nn.Linear(self.fg, 256)
#         self.linear_x_2 = nn.Linear(256, 128)
#         self.linear_x_3 = nn.Linear(128, 64)
#
#         self.linear_y_1 = nn.Linear(self.fd, 256)
#         self.linear_y_2 = nn.Linear(256, 128)
#         self.linear_y_3 = nn.Linear(128, 64)
#
#     def forward(self, input):
#         torch.manual_seed(1)
#         x_m = torch.randn(self.m, self.fg)  # x_m(495, 256)
#         x_d = torch.randn(self.d, self.fd)  # x_d(383, 256)
#
#         X1 = torch.relu(self.gcn_x1(x_m.cuda(), input[1]['edge_index'].cuda(),
#                                     input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
#         X = torch.relu(self.gcn_x2(X1, input[1]['edge_index'].cuda(),
#                                    input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
#         # X(495, 256), X1(495, 256)
#
#         Y1 = torch.relu(self.gcn_y1(x_d.cuda(), input[0]['edge_index'].cuda(),
#                                     input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
#         Y = torch.relu(self.gcn_y2(Y1, input[0]['edge_index'].cuda(),
#                                    input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
#         # Y(383, 256), Y1(383, 256)
#
#         x1 = torch.relu(self.linear_x_1(X))
#         x2 = torch.relu(self.linear_x_2(x1))
#         x = torch.relu(self.linear_x_3(x2))
#         # x(495, 64)
#
#         y1 = torch.relu(self.linear_y_1(Y))
#         y2 = torch.relu(self.linear_y_2(y1))
#         y = torch.relu(self.linear_y_3(y2))
#         # y(383, 64)
#
#         return x.mm(y.t())
from HGNN.models import HGNN_conv


class Model(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(Model, self).__init__()

        self.dropout = dropout
        self.hgc_x1 = HGNN_conv(in_ch, n_hid)
        self.hgc_x2 = HGNN_conv(n_hid, n_class)

        self.hgc_y1 = HGNN_conv(in_ch, n_hid)
        self.hgc_y2 = HGNN_conv(n_hid, n_class)

        # super(Model, self).__init__()

        # self.fg = sizes.fg
        # self.fd = sizes.fd
        # self.k = sizes.k
        # self.m = sizes.m
        # self.d = sizes.d
        # self.gcn_x1 = conv.GCNConv(self.fg, self.fg)
        # self.gcn_y1 = conv.GCNConv(self.fd, self.fd)
        # self.gcn_x2 = conv.GCNConv(self.fg, self.fg)
        # self.gcn_y2 = conv.GCNConv(self.fd, self.fd)

        # self.linear_gcn_x1 = nn.Linear(self.fg, self.fg)
        # self.linear_gcn_y1 = nn.Linear(self.fd, self.fd)
        # self.linear_gcn_x2 = nn.Linear(self.fg, self.fg)
        # self.linear_gcn_y2 = nn.Linear(self.fd, self.fd)

        # self.linear_x_1 = nn.Linear(self.fg, 256)
        # self.linear_x_2 = nn.Linear(256, 128)
        # self.linear_x_3 = nn.Linear(128, 64)
        #
        # self.linear_y_1 = nn.Linear(self.fd, 256)
        # self.linear_y_2 = nn.Linear(256, 128)
        # self.linear_y_3 = nn.Linear(128, 64)

    # def forward(self, input):
    # torch.manual_seed(1)
    # x_m = torch.randn(self.m, self.fg)  # x_m(495, 256)
    # x_d = torch.randn(self.d, self.fd)  # x_d(383, 256)
    #
    # X1 = torch.relu(self.gcn_x1(x_m.cuda(), input[1]['edge_index'].cuda(),
    #                             input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
    # X = torch.relu(self.gcn_x2(X1, input[1]['edge_index'].cuda(),
    #                            input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
    # # X(495, 256), X1(495, 256)
    #
    # Y1 = torch.relu(self.gcn_y1(x_d.cuda(), input[0]['edge_index'].cuda(),
    #                             input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
    # Y = torch.relu(self.gcn_y2(Y1, input[0]['edge_index'].cuda(),
    #                            input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))
    # # Y(383, 256), Y1(383, 256)
    #
    # x1 = torch.relu(self.linear_x_1(X))
    # x2 = torch.relu(self.linear_x_2(x1))
    # x = torch.relu(self.linear_x_3(x2))
    # # x(495, 64)
    #
    # y1 = torch.relu(self.linear_y_1(Y))
    # y2 = torch.relu(self.linear_y_2(y1))
    # y = torch.relu(self.linear_y_3(y2))
    # # y(383, 64)
    def forward(self, x, G):
        # X1 = torch.relu(self.gcn_x1(x_m.cuda(), input[1]['edge_index'].cuda(),
        #                             input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)

        return x.mm(y.t())


