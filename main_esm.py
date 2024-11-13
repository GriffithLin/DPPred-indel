#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/7 21:27
# @Author  : zdj
# @FileName: main_test.py.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 8:54
# @Author : fhh
# @FileName: main.py
# @Software: PyCharm

import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from train import predict, CosineScheduler, DataTrain_confusion, predict_confusion
from my_util import get_config, data_load_npy_confusion_cont, data_load_npy, save_results, print_score, spent_time, \
    data_load_npy_confusion_predict, data_load_npy_predict
from models.model import TextCNN, TextCNN_CLS, TextCNN_confusion, TextCNN_confusion_noFan, TextCNN_confusion_Contrastive, ContrastiveLoss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import estimate

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(DEVICE)

bases_kmer = 'ATCG'
bases = 'XATCG'

#
# def collate_cont(batch):
#     half_batch_size = int(len(batch)/2)
#     dna1_list = []
#     dna2_list = []
#     protein1_list = []
#     protein2_list = []
#     label_list = []
#     label1_list = []
#     label2_list = []
#     for i in range(half_batch_size):
#         j = i + half_batch_size
#         dna1, protein1, label1 = batch[i][0], batch[i][1], batch[i][2]
#         dna2, protein2, label2 = batch[j][0], batch[j][1], batch[j][2]
#         dna1_list.append(dna1)
#         dna2_list.append(dna2)
#         protein1_list.append(protein1)
#         protein2_list.append(protein2)
#         label1_list.append(int(label1))
#         label2_list.append(int(label2))
#         label = (int(label1) ^ int(label2))
#         label_list.append(label)
#     dna1_list = torch.from_numpy(np.asarray(dna1_list))
#     dna2_list = torch.from_numpy(np.asarray(dna2_list))
#     protein1_list = torch.from_numpy(np.asarray(protein1_list))
#     protein2_list = torch.from_numpy(np.asarray(protein2_list))
#     label1_list = torch.from_numpy(np.asarray(label1_list))
#     label2_list = torch.from_numpy(np.asarray(label2_list))
#     label_list = torch.from_numpy(np.asarray(label_list))
#     return  dna1_list, dna2_list, protein1_list, protein2_list, label1_list, label2_list, label_list
#
#
# def mark_label(src):
#     assert src.find("hgmd") >= 0 or src.find("pos") >= 0 or src.find("gnomAD") >= 0 or src.find("neg") >= 0 or src.find(
#         "Neg") >= 0 or src.find("Pos") >= 0
#
#     if src.find("hgmd") >= 0 or src.find("pos") >= 0 or src.find("Pos") >= 0:
#         return 1
#     return 0
#
#
# class NpyDataset(Dataset):
#     def __init__(self, npy_file, label_file):
#         self.labels = np.load(label_file)
#         if npy_file.find("dna") != -1:
#             embedding_size = 768
#             seq_len = 200
#         else:
#             embedding_size = 1280
#             seq_len = 1024
#         self.data = np.memmap(npy_file, dtype=np.float32, mode='r',
#                               shape=(self.labels.shape[0], seq_len, embedding_size))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])
#         return (self.data[idx]), (self.labels[idx])
#
#
# class NpyDataset_confusion(Dataset):
#     def __init__(self, protein_file, dna_file, label_file):
#         self.data_dna = np.lib.format.open_memmap(dna_file)
#         self.data_label = np.load(label_file)
#         self.data_protein = np.memmap(protein_file, dtype=np.float32, mode="r", shape=(self.data_label.shape[0], 1024, 1280))
#
#     def __len__(self):
#         return len(self.data_dna)
#
#     def __getitem__(self, idx):
#         return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx])
#
#
# def data_load_npy(data_path, train_path, test_path, batch=32):
#     train_label = os.path.join(data_path, "train_labels.npy")
#     dataset_train = NpyDataset(train_path, train_label)
#     dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#     #
#     test_label = os.path.join(data_path, "DDD_labels.npy")
#     dataset_test = NpyDataset(test_path, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_train, dataset_test
#
#
# def data_load_npy_confusion(data_path, batch=32):
#     train_label = os.path.join(data_path, "train_labels.npy")
#     train_protein_data = os.path.join(data_path, "train.npy")
#     train_dna_data = os.path.join(data_path, "train_dna.npy")
#     dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
#     dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#     #
#     test_label = os.path.join(data_path, "DDD_labels.npy")
#     test_protein_data = os.path.join(data_path, "DDD.npy")
#     test_dna_data = os.path.join(data_path, "DDD_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_train, dataset_test
#
# def data_load_npy_confusion_cont(data_path, dna_data_direction, batch=32):
#     train_label = os.path.join(data_path, "train_labels.npy")
#     train_protein_data = os.path.join(data_path, "train.npy")
#     train_dna_data = os.path.join(dna_data_direction, "train_dna.npy")
#     dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
#     dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#     dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn = collate_cont, drop_last= False)
#     #
#     test_label = os.path.join(data_path, "test_labels.npy")
#     test_protein_data = os.path.join(data_path, "test.npy")
#     test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_loader_train, dataset_loader_train_cont, dataset_test
#
#
# def data_load_npy_predict(test_path, test_label, batch=32):
#     dataset_test = NpyDataset(test_path, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_test
#
#
# def data_load_npy_confusion_predict(data_path, dna_data_direction, batch=32):
#     test_label = os.path.join(data_path, "DDD_labels.npy")
#     test_protein_data = os.path.join(data_path, "DDD.npy")
#     test_dna_data = os.path.join(dna_data_direction, "DDD_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_test



def train(args, paths=None):
    start_time = time.time()
    file_path = "{}/{}.csv".format('result', 'test')  # 结果保存路径

    print("Data is loading......（￣︶￣）↗　")

    if args.confusion == True:
        train_dataset, dataset_train_cont, test_dataset = data_load_npy_confusion_cont(args.data_direction, args.dna_data_direction, args.batch_size)
    else:
        train_dataset, test_dataset = data_load_npy(args.data_direction, args.train_direction, args.test_direction,
                                                    args.batch_size)
    print("Data is loaded!�?≧▽�?)o")

    all_test_score = 0  # 初始话评估指�?
    # 训练并保存模�?
    print(f"{args.model_name} is training......")
    for counter in range(args.model_num):
        train_start = time.time()
        if args.Contrastive:
            model = TextCNN_confusion_Contrastive(args.filter_num, args.filter_size,
                                      args.output_size, args.dropout)
        elif args.confusion:
            model = TextCNN_confusion(args.filter_num, args.filter_size_dna, args.filter_size_protein,
                                      args.output_size, args.dropout)
        else:
            if args.train_direction.find("dna") != -1:
                model = TextCNN_CLS(args.filter_num, args.filter_size,
                                    args.output_size, args.dropout, 768)
            else:
                model = TextCNN_CLS(args.filter_num, args.filter_size,
                                    args.output_size, args.dropout, 1280)
        print(model)

        if args.opt == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True)  # 优化�?
            print(args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 优化�?
        lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500
                                       )  # 退化学习率
        criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
        criterion_cont = ContrastiveLoss()
        # 初始化训练类
        Train = DataTrain_confusion(model, optimizer, criterion, criterion_cont, lr_scheduler, device=DEVICE)


        # 训练模型
        # if args.divide_validata:
        #     Train.train_step_val(train_dataset, val_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter, early_stop=args.early_stop, threshold = args.threshold)
        # else:
        if args.Contrastive:
            Train.train_step_cont(train_dataset, dataset_train_cont, test_dataset, args.model_name, epochs=args.epochs, model_num=counter,
                             early_stop=args.early_stop, threshold=args.threshold)
        else:
            Train.train_step(train_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter,
                             early_stop=args.early_stop, threshold=args.threshold)

        # 保存模型
        PATH = os.getcwd()
        each_model = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
        torch.save(model.state_dict(), each_model)

        # 模型预测
        if args.confusion == True:
            model_predictions, true_labels = predict_confusion(model, test_dataset, device=DEVICE)
        else:
            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
        # 模型评估
        test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)

        # 保存评估结果
        train_end = time.time()
        # if len(train_datasets)>1:
        #     save_results(parse.model_name + "fold " + str(i), train_start, train_end, test_score, file_path)
        # else:
        save_results(parse.model_name, train_start, train_end, test_score, file_path)

        # 打印评估结果
        print(f"{args.model_name}:{counter + 1}")
        print("测试集：")
        print_score(test_score)
        df_test_score = pd.DataFrame(test_score, index=[0])
        if type(all_test_score) == int:
            all_test_score = df_test_score
        else:
            all_test_score = all_test_score + df_test_score

    "-------------------------------------------打印平均结果-----------------------------------------------"
    run_time = time.time()
    m, s = spent_time(start_time, run_time)  # 运行时间
    print(f"runtime:{m}m{s}s")
    print("测试集：")
    all_test_score = all_test_score / (args.model_num)
    print_score(all_test_score)
    save_results('average', start_time, run_time, all_test_score, file_path)
    "---------------------------------------------------------------------------------------------------"


def do_predict(args):
    file_path = "{}/{}.csv".format('result', 'test')  # 评价指标保存

    print("data loading....")
    if args.confusion:
        test_dataset = data_load_npy_confusion_predict(args.data_direction, args.dna_data_direction, args.batch_size)
    else:
        test_label = "/data3/linming/DNA_Lin/esm/scripts/data/" + "DDD_labels.npy"

        test_dataset = data_load_npy_predict(args.test_DDD_direction, test_label,  args.batch_size)

    if args.Contrastive:
        model = TextCNN_confusion_Contrastive(args.filter_num, args.filter_size,
                                              args.output_size, args.dropout)
    elif args.confusion:
        model = TextCNN_confusion(args.filter_num, args.filter_size_dna, args.filter_size_protein,
                                      args.output_size, args.dropout)
    else:
        if args.test_direction.find("dna") != -1:
            model = TextCNN_CLS(args.filter_num, args.filter_size,
                                args.output_size, args.dropout, 768)
        else:
            model = TextCNN_CLS(args.filter_num, args.filter_size,
                                args.output_size, args.dropout, 1280)
    print(model)

    model.load_state_dict(torch.load(args.model_path))
    # 模型预测

    if args.confusion:
        model_predictions, true_labels = predict_confusion(model, test_dataset, device=DEVICE)
    else:
        model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
    # 模型评估
    test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)

    save_results(parse.model_name, 0, 0, test_score, file_path)


if __name__ == '__main__':
    parse = get_config()  # 获取参数
    print(parse)
    # parse.model_name = 'tc_cbam_drop0.1'
    # parse.model_num = 1
    parse.threshold = 0.5
    parse.confusion = True
    parse.epochs = 60
    parse.dropout = 0.7
    parse.filter_size_dna = [3, 4, 5, 8, 16, 32]
    parse.filter_size = parse.filter_size_dna
    parse.filter_size_protein = [3, 4, 5, 8, 16, 32]
    parse.model_name = "textCNN_dropout0.7_60"
    path = []
    parse.do_train = True
    parse.do_predict = True

    parse.Contrastive = False
    parse.early_stop = 10

    k_mer = 5
    crop_len = 225
    parse.dna_data_direction = os.path.join(parse.dna_data_direction , str(k_mer), str(crop_len))
    parse.model_name = parse.model_name + str(k_mer) + str(crop_len)
    # parse.data_direction = "/data3/linming/DNA_Lin/esm/scripts/data/"
    # # parse.train_direction = "/data3/linming/DNA_Lin/esm/scripts/data/train.npy"
    # # parse.test_direction = "/data3/linming/DNA_Lin/esm/scripts/data/test.npy"
    # # parse.test_DDD_direction = "/data3/linming/DNA_Lin/esm/scripts/data/DDD.npy"
    #
    # parse.train_direction = os.path.join(parse.dna_data_direction, "train_dna.npy")
    # parse.test_direction = os.path.join(parse.dna_data_direction, "test_dna.npy")
    # parse.test_DDD_direction = os.path.join(parse.dna_data_direction, "DDD", "DDD_dna.npy")
    if parse.confusion:
        parse.model_name = parse.model_name + "_confusion"
    else:
        if parse.test_direction.find("dna") != -1:
            parse.model_name = parse.model_name + "_dna"
        else:
            parse.model_name = parse.model_name + "_protein"

    if parse.Contrastive:
        parse.model_name = parse.model_name + "_Cont"

    for num in range(parse.model_num):
        a = f'saved_models/tc_cbam{num}.pth'
        path.append(a)

    #
    if parse.do_train:
        train(parse)
    if parse.do_predict:
        PATH = os.getcwd()
        # parse.model_name = "DPPred-indel"
        each_model = os.path.join(PATH, 'saved_models', parse.model_name + '.pth')
        parse.model_path = each_model
        parse.model_name = parse.model_name + "_test"
        do_predict(parse)

    # parse.epochs = 63
    # parse.filter_size_dna = [3]
    # parse.filter_size_protein = [3]
    # parse.model_name = parse.model_name + "_CNN_63"
    # if parse.do_train:
    #     train(parse)
