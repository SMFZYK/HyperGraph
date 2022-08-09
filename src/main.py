# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.16
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from torch import optim, nn, save
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_feature_construct_H
from models import HGNN
from trainData import Dataset
from prepareData import prepare_data
from utils import hypergraph_utils as hgut
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32


class Config(object):
    def __init__(self):
        self.data_path = '../datasets/MDAData/data(383-495)'
        self.validation = 5
        self.save_path = '../save'
        self.epoch = 1000
        self.alpha = 0.1


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1 - opt.alpha) * loss_sum[one_index].sum() + opt.alpha * loss_sum[zero_index].sum()


def train(model, train_data, optimizer, opt, G1_Sum, G2_Sum, cha_index, cha_index0, F1_score, AUC_ROC, AUPR, EPOCH, SUM):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()

    # fts1, H1 = \
    #     load_feature_construct_H('../dataset/data(383-495)/m-m.csv')
    # G1 = hgut.generate_G_from_H(H1, variable_weight=False)
    #
    # fts2, H2 = \
    #     load_feature_construct_H('../dataset/data(383-495)/d-d.csv')
    # G2 = hgut.generate_G_from_H(H2, variable_weight=False)

    # train_data[0]['data'] = torch.Tensor(train_data[0]['data']).to(device)
    # train_data[1]['data'] = torch.Tensor(train_data[1]['data']).to(device)
    # G1 = torch.Tensor(G1).to(device)
    # G2 = torch.Tensor(G2).to(device)
    def train_epoch():
        model.zero_grad()
        # train_data = torch.Tensor(train_data).to(device)
        scores = model(train_data, G1_Sum, G2_Sum)  # score(495, 383)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), scores)
        loss.backward()
        optimizer.step()
        return loss, scores

    for epoch in range(1, opt.epoch + 1):
        if epoch > 800:
            if SUM[-1] <= SUM[-2]:
                save(model.state_dict(), '../save/HGNN_MDA_all5.pt')
            else:
                pass
        if epoch == opt.epoch:
            vision(EPOCH, F1_score, AUC_ROC, AUPR, SUM)
        train_reg_loss, score = train_epoch()
        if epoch % 10 == 0:
            print(train_reg_loss.item() / (len(one_index[0]) + len(zero_index[0])))
        if epoch % 10 == 0:
            # for i in range(score.size(0)):
            #     for j in range(score.size(1)):
            #         if score[i][j] < 0.5:
            #             # zero_index.append([i, j])
            #             score[i][j] = 0
            #         if score[i][j] >= 0.5:
            #             score[i][j] = 1

            # print("epoch: ", epoch)
            # print(score)
            # scoreTP = scoreFP = scoreFN = scoreTN = 0
            # a = score[cha_index]
            # dataPre = []
            # dataAct = []
            # for ind in cha_index:
            #     dataAct.append(1)
            #     if score[ind[0], ind[1]] == 0:
            #         scoreFN += 1
            #         dataPre.append(0)
            #     else:
            #         scoreTP += 1
            #         dataPre.append(1)
            # for ind0 in cha_index0:
            #     dataAct.append(0)
            #     if score[ind0[0], ind0[1]] == 1:
            #         scoreFP += 1
            #         dataPre.append(1)
            #     else:
            #         scoreTN += 1
            #         dataPre.append(0)
            #
            # # labels = [0, 1]
            # cm = confusion_matrix(dataAct, dataPre)
            # print(cm)
            # sns.heatmap(cm, annot=True, annot_kws={'size': 20, 'weight': 'bold', 'color': 'blue'})
            # plt.rc('font', family='Arial Unicode MS', size=14)
            # plt.title('混淆矩阵', fontsize=20)
            # plt.xlabel('Predict', fontsize=14)
            # plt.ylabel('Actual', fontsize=14)
            # plt.show()
            # print("a = ", a)
            # print("FN: {}, TP: {}, FP: {}, TN: {}".format(scoreFN / len(cha_index), scoreTP / len(cha_index), scoreFP / len(cha_index0), scoreTN / len(cha_index0)))
            # # FPR, TPR, threshold = roc_curve(data[:, 1], data[:, 2], pos_label=1)
            #
            # # 精确率Precision
            # P = scoreTP / (scoreTP + scoreFP)
            # # 召回率Recall
            # R = scoreTP / (scoreTP + scoreFN)
            # # F1
            # F1 = 2 / (1 / P + 1 / R)
            # # 准确率Accuracy
            # Acc = (scoreTP + scoreTN) / (scoreTP + scoreFP + scoreFN + scoreTN)

            dataPre = []
            dataAct = []
            for ind in cha_index:
                dataAct.append(1)
                dataPre.append(score[ind[0], ind[1]].data.cpu().numpy())
            for ind0 in cha_index0:
                dataAct.append(0)
                dataPre.append(score[ind0[0], ind0[1]].data.cpu().numpy())
            # 绘制ROC/AUC
            act = np.array(dataAct)
            pre = np.array(dataPre)
            FPR, TPR, thresholds = roc_curve(act, pre)
            AUC = auc(FPR, TPR)
            print('AUC:', AUC)
            # plt.rc('font', family='Arial Unicode MS', size=14)
            # plt.plot(FPR, TPR, label="AUC={:.2f}".format(AUC), marker='o', color='b', linestyle='--')
            # plt.legend(loc=4, fontsize=10)
            # plt.title('ROC曲线', fontsize=20)
            # plt.xlabel('FPR', fontsize=14)
            # plt.ylabel('TPR', fontsize=14)
            # plt.show()

            # print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}".format(P, R, F1, Acc))

            # 严格定义计算方法
            precision, recall, thresholds = precision_recall_curve(dataAct, dataPre)
            PR = auc(recall, precision)
            print("PR: ", PR)
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # # plt.grid()  # 生成网格
            #
            # plt.plot(recall, precision)
            # plt.figure("P-R Curve")
            # plt.show()

            dataPre = np.around(dataPre, 0).astype(int)
            f1_weighted = f1_score(dataAct, dataPre, average='weighted')
            f1_macro = f1_score(dataAct, dataPre, average='macro')
            print("f1-score: 考虑类别的不平衡性为{}, 不考虑类别的不平衡性为{}".format(f1_weighted, f1_macro))

            Sum = f1_weighted + AUC + PR
            Sum = Sum / 3

            F1_score.append(f1_weighted)
            AUC_ROC.append(AUC)
            AUPR.append(PR)
            EPOCH.append(epoch)
            SUM.append(Sum)
            # print()
            pass
        pass


def vision(EPOCH, F1_score, AUC_ROC, AUPR, SUM):
    # 绘制三指标随epochs变化图像
    # plt.plot(EPOCH, F1_score)
    # plt.xlabel('epochs')
    # plt.ylabel('F1_score')
    # plt.title('F1随epochs变化曲线', fontsize=20)
    # plt.show()
    # plt.plot(EPOCH, AUC_ROC)
    # plt.xlabel('epochs')
    # plt.ylabel('AUC_ROC')
    # plt.title('AUC_ROC随epochs变化曲线', fontsize=20)
    # plt.show()
    # plt.plot(EPOCH, AUPR)
    # plt.xlabel('epochs')
    # plt.ylabel('AUPR')
    # plt.title('AUPR随epochs变化曲线', fontsize=20)
    # plt.show()
    plt.plot(EPOCH, SUM)
    plt.xlabel('epochs')
    plt.ylabel('SUM')
    plt.title('SUM随epochs变化曲线', fontsize=20)
    plt.show()


opt = Config()


def main():
    dataset, cha_index, cha_index0, cha_index1, cha_index2 = prepare_data(opt)
    # sizes = Sizes(dataset)
    train_data = Dataset(opt, dataset)
    G1_Sum = {}
    G2_Sum = {}
    for i in range(1, 8):
        fts1, H1 = \
            load_feature_construct_H('../datasets/MDAData/data(383-495)/m-m.csv', K_neigs=[i])
        G1 = hgut.generate_G_from_H(H1, variable_weight=False)

        fts2, H2 = \
            load_feature_construct_H('../datasets/MDAData/data(383-495)/d-d.csv', K_neigs=[i])
        G2 = hgut.generate_G_from_H(H2, variable_weight=False)
        G1 = torch.Tensor(G1).to(device)
        G2 = torch.Tensor(G2).to(device)
        G1_Sum[i] = G1
        G2_Sum[i] = G2
    model1 = HGNN()
    model1 = model1.to(device)
    train_data[0][0]['data'] = torch.Tensor(train_data[0][0]['data']).to(device)
    train_data[0][1]['data'] = torch.Tensor(train_data[0][1]['data']).to(device)
    F1_score1 = []
    AUC_ROC1 = []
    AUPR1 = []
    EPOCH1 = []
    SUM1 = []

    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    train(model1, train_data[1], optimizer, opt, G1_Sum, G2_Sum, cha_index=cha_index, cha_index0=cha_index0,
          F1_score=F1_score1, AUC_ROC=AUC_ROC1, AUPR=AUPR1, EPOCH=EPOCH1, SUM=SUM1)

    F1_score = []
    AUC_ROC = []
    AUPR = []
    SUM = []
    for i in range(1, opt.validation + 1):
        print('-' * 50)
        print("Training, {} dataset".format(i))
        # model = Model(sizes)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # train_data[i][0]['data'] = torch.Tensor(train_data[i][0]['data']).to(device)
        # train_data[i][1]['data'] = torch.Tensor(train_data[i][1]['data']).to(device)
        # train(model, train_data[i], optimizer, opt, G1_Sum, G2_Sum, cha_index=cha_index, cha_index0=cha_index0,
        #       F1_score=F1_score, AUC_ROC=AUC_ROC, AUPR=AUPR, EPOCH=EPOCH, SUM=SUM)
        # print(G2)
        # print(G1)
        model = HGNN()
        model = model.to(device)
        m_state_dict = torch.load('../save/HGNN_MDA_all5.pt')
        model.load_state_dict(m_state_dict)
        scores = model(train_data[i], G1_Sum, G2_Sum)
        dataPre = []
        dataAct = []
        for ind in cha_index1:
            dataAct.append(1)
            dataPre = np.append(dataPre, scores[ind[0], ind[1]].data.cpu().numpy())
        for ind0 in cha_index2:
            dataAct.append(0)
            dataPre = np.append(dataPre, scores[ind0[0], ind0[1]].data.cpu().numpy())
        # 绘制ROC/AUC
        act = np.array(dataAct)
        pre = np.array(dataPre)
        FPR, TPR, thresholds = roc_curve(act, pre)
        AUC = auc(FPR, TPR)
        print('AUC:', AUC)
        plt.rc('font', family='Arial Unicode MS', size=14)
        plt.plot(FPR, TPR, label="AUC={:.2f}".format(AUC), marker='o', color='b', linestyle='--')
        plt.legend(loc=4, fontsize=10)
        plt.title('ROC曲线', fontsize=20)
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.show()

        # print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}".format(P, R, F1, Acc))

        # 严格定义计算方法
        precision, recall, thresholds = precision_recall_curve(dataAct, dataPre)
        PR = auc(recall, precision)
        print("PR: ", PR)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.grid()  # 生成网格

        plt.plot(recall, precision)
        plt.figure("P-R Curve")
        plt.show()

        dataPre = np.around(dataPre, 0).astype(int)
        f1_weighted = f1_score(dataAct, dataPre, average='weighted')
        f1_macro = f1_score(dataAct, dataPre, average='macro')
        print("f1-score: 考虑类别的不平衡性为{}, 不考虑类别的不平衡性为{}".format(f1_weighted, f1_macro))

        F1_score.append(f1_weighted)
        AUC_ROC.append(AUC)
        AUPR.append(PR)
        Sum = (f1_weighted + PR + AUC) / 3
        SUM = np.append(SUM, Sum)
        pass
    print('F1_score: {}, AUC_ROC: {}, AUPR: {}, SUM: {}'.format(np.mean(F1_score), np.mean(AUC_ROC),
                                                                np.mean(AUPR), np.mean(SUM)))


if __name__ == '__main__':
    main()
