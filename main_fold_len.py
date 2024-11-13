#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/16 19:24
# @Author  : zdj
# @FileName: main_fold.py.py
# @Software: PyCharm
# !/usr/bin/env python
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
from my_util import get_config
from models.model import TextCNN, TextCNN_confusion, TextCNN_CLS, TextCNN_confusion_Contrastive, ContrastiveLoss, TextCNN_confusion_batch_pad
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import estimate

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(DEVICE)

bases_kmer = 'ATCG'
bases = 'XATCG'

#
# def collate_cont(batch):
#     half_batch_size = int(len(batch) / 2)
#     dna1_list = []
#     dna2_list = []
#     protein1_list = []
#     protein2_list = []
#     label_list = []
#     label1_list = []
#     label2_list = []
#     protein_len_list1 = []
#     protein_len_list2 = []
#     for i in range(half_batch_size):
#         j = i + half_batch_size
#         dna1, protein1, label1, protein_len1 = batch[i][0], batch[i][1], batch[i][2], batch[i][3]
#         dna2, protein2, label2, protein_len2 = batch[j][0], batch[j][1], batch[j][2], batch[j][3]
#         dna1_list.append(dna1)
#         dna2_list.append(dna2)
#         protein1_list.append(protein1)
#         protein2_list.append(protein2)
#         label1_list.append(int(label1))
#         label2_list.append(int(label2))
#         label = (int(label1) ^ int(label2))
#         label_list.append(label)
#         protein_len_list1.append(protein1)
#         protein_len_list2.append(protein2)
#     dna1_list = torch.from_numpy(np.asarray(dna1_list))
#     dna2_list = torch.from_numpy(np.asarray(dna2_list))
#     protein1_list = torch.from_numpy(np.asarray(protein1_list))
#     protein2_list = torch.from_numpy(np.asarray(protein2_list))
#     label1_list = torch.from_numpy(np.asarray(label1_list))
#     label2_list = torch.from_numpy(np.asarray(label2_list))
#     label_list = torch.from_numpy(np.asarray(label_list))
#     protein_len_list1 = torch.from_numpy(np.asarray(protein_len_list1))
#     protein_len_list2 = torch.from_numpy(np.asarray(protein_len_list2))
#     return dna1_list, dna2_list, protein1_list, protein2_list, label1_list, label2_list, label_list, protein_len_list1, protein_len_list2
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
#     def __init__(self, protein_file, dna_file, label_file, protein_len_file):
#         self.data_dna = np.lib.format.open_memmap(dna_file)
#         self.data_label = np.load(label_file)
#         self.data_protein = np.memmap(protein_file, dtype=np.float32, mode="r",
#                                       shape=(self.data_label.shape[0], 1024, 1280))
#         self.data_proten_len = pd.read_csv(protein_len_file)["strlen"]
#
#     def __len__(self):
#         return len(self.data_dna)
#
#     def __getitem__(self, idx):
#         return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx]), (self.data_proten_len[idx])
#
#
# class NpyDataset_confusion_div(Dataset):
#     def __init__(self, data_protein, data_dna, data_label, data_protein_len):
#         self.data_dna = data_dna
#         self.data_label = data_label
#         self.data_protein = data_protein
#         self.protein_len = data_protein_len
#
#     def __len__(self):
#         return len(self.data_dna)
#
#     def __getitem__(self, idx):
#         return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx]), (self.protein_len[idx])
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
#
# def data_load_npy_confusion_cont(data_path, dna_data_direction, batch=32):
#     train_label = os.path.join(data_path, "train_labels.npy")
#     train_protein_data = os.path.join(data_path, "train.npy")
#     train_dna_data = os.path.join(dna_data_direction, "train_dna.npy")
#     dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
#     dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#     dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
#                                            drop_last=False)
#     #
#     test_label = os.path.join(data_path, "test_labels.npy")
#     test_protein_data = os.path.join(data_path, "test.npy")
#     test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_loader_train, dataset_loader_train_cont, dataset_test
#
# # todo: add strlen for cont
# def data_load_npy_confusion_cont_k_fold(data_path, dna_data_direction, k_fold, batch=32):
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     data_label_path = os.path.join(data_path, "train_labels.npy")
#     data_protein_path = os.path.join(data_path, "train.npy")
#     data_dna_path = os.path.join(dna_data_direction, "train_dna.npy")
#     data_protein_len_path = os.path.join("/data3/linming/DNA_Lin/dataCenter/", "train_list.csv")
#
#     data_label = np.load(data_label_path)
#     data_dna = np.lib.format.open_memmap(data_dna_path)
#     data_protein = np.memmap(data_protein_path, dtype=np.float32, mode="r",
#                              shape=(data_label.shape[0], 1024, 1280))
#     data_protein_len = pd.read_csv(data_protein_len_path)["strlen"]
#     datasets_train, datasets_dev, datasets_cont = [], [], []
#     for i, (train_index, dev_index) in enumerate(cv.split(data_dna, data_label)):
#         if i != k_fold:
#             continue
#         train_protein_data, train_dna_data, train_label, train_protein_len = data_protein[train_index], data_dna[train_index], data_label[
#             train_index], data_protein_len[train_index]
#         dataset_train = NpyDataset_confusion_div(train_protein_data, train_dna_data, train_label, train_protein_len)
#         dev_protein_data, dev_dna_data, dev_label, dev_protein_len = data_protein[dev_index], data_dna[dev_index], data_label[
#             dev_index], data_protein_len[dev_index]
#         dataset_dev = NpyDataset_confusion_div(dev_protein_data, dev_dna_data, dev_label, dev_protein_len)
#
#         dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#         dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
#                                                drop_last=False)
#         dataset_loader_dev = DataLoader(dataset_dev, batch_size=batch, shuffle=True)
#
#         datasets_train.append(dataset_loader_train)
#         datasets_cont.append(dataset_loader_train_cont)
#         datasets_dev.append(dataset_loader_dev)
#
#     # test data
#     test_label = os.path.join(data_path, "test_labels.npy")
#     test_protein_data = os.path.join(data_path, "test.npy")
#     test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
#     test_protein_len_path = os.path.join("/data3/linming/DNA_Lin/dataCenter/", "test_list.csv")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label, test_protein_len_path)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return datasets_train, datasets_cont, datasets_dev, dataset_test
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
#     test_label = os.path.join(data_path, "test_labels.npy")
#     test_protein_data = os.path.join(data_path, "test.npy")
#     test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_test
#
#
# def spent_time(start, end):
#     epoch_time = end - start
#     minute = int(epoch_time / 60)  # 分钟
#     secs = int(epoch_time - minute * 60)  # �?
#     return minute, secs
#
#
# # '%.3f' % test_score[5],
# def save_results(model_name, start, end, test_score, file_path):
#     # 保存模型结果 csv文件
#     title = ['Model']
#     title.extend(test_score.keys())
#     title.extend(['RunTime', 'Test_Time'])
#
#     now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     content_row = [model_name]
#     for key in test_score:
#         content_row.append('%.3f' % test_score[key])
#     content_row.extend([[end - start], now])
#
#     content = [content_row]
#
#     if os.path.exists(file_path):
#         data = pd.read_csv(file_path, header=None)
#         one_line = list(data.iloc[0])
#         if one_line == title:
#             with open(file_path, 'a+', newline='') as t:  # newline用来控制空的行数
#                 writer = csv.writer(t)  # 创建一个csv的写入器
#                 writer.writerows(content)  # 写入数据
#         else:
#             with open(file_path, 'a+', newline='') as t:
#                 writer = csv.writer(t)
#                 writer.writerow(title)  # 写入标题
#                 writer.writerows(content)
#     else:
#         with open(file_path, 'a+', newline='') as t:
#             writer = csv.writer(t)
#             writer.writerow(title)
#             writer.writerows(content)
#
#
# def print_score(test_score):
#     if type(test_score) == dict:
#         for key in test_score:
#             print(key + f': {test_score[key]:.3f}')
#     else:
#         print(test_score)


def train(args, paths=None):
    start_time = time.time()
    file_path = "{}/{}.csv".format('result', 'test')  # 结果保存路径

    print("Data is loading......（￣︶￣）↗　")
    # if args.k_fold != 1:

    if args.confusion == True:
        train_datasets, train_datasets_cont, dev_datasets, test_dataset2 = data_load_npy_confusion_cont_k_fold(
            args.data_direction, args.dna_data_direction, args.k_fold, args.batch_size)
    else:
        train_dataset, test_dataset = data_load_npy(args.data_direction, args.train_direction, args.test_direction,
                                                    args.batch_size)
    print("Data is loaded!�?≧▽�?)o")

    all_test_score = 0  # 初始话评估指�?
    # 训练并保存模�?
    counter = 1
    print(f"{args.model_name} is training......")
    i = 1
    for train_dataset, train_dataset_cont, test_dataset in zip(train_datasets, train_datasets_cont, dev_datasets):
        train_start = time.time()
        if args.Contrastive:
            model = TextCNN_confusion_Contrastive(args.filter_num, args.filter_size,
                                                  args.output_size, args.dropout)
        elif args.confusion:
            model = TextCNN_confusion_batch_pad(args.filter_num, args.filter_size,
                                      args.output_size, args.dropout)
        else:
            model = TextCNN_CLS(args.filter_num, args.filter_size,
                                args.output_size, args.dropout, 1280)
        print(model)

        if args.opt == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True)  # 优化�?
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 优化�?
        lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500
                                       )  # 退化学习率
        criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
        criterion_cont = ContrastiveLoss()
        # 初始化训练类
        Train = DataTrain_confusion(model, optimizer, criterion, criterion_cont, lr_scheduler, device=DEVICE)

        # 训练模型
        # if args.divide_validata and args.Contrastive:
        #     Train.train_step_cont(train_dataset, train_dataset_cont, test_dataset, args.model_name, epochs=args.epochs, model_num=counter,
        #                           early_stop=args.early_stop, threshold = args.threshold)

        if args.Contrastive:
            Train.train_step_cont(train_dataset, train_dataset_cont, test_dataset, args.model_name, epochs=args.epochs,
                                  model_num=counter,
                                  early_stop=args.early_stop, threshold=args.threshold)
        else:
            Train.train_step(train_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter,
                             early_stop=args.early_stop, threshold=args.threshold)

        # 保存模型
        PATH = os.getcwd()
        each_model = os.path.join(PATH, 'saved_models', args.model_name + str(counter) + '.pth')
        torch.save(model.state_dict(), each_model)

        # 模型预测
        if args.confusion == True:
            model_predictions, true_labels = predict_confusion(model, test_dataset, device=DEVICE)
        # else:
        #     model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)

        # 模型评估
        test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)

        # 保存评估结果
        train_end = time.time()
        if len(train_datasets) > 1:
            save_results(parse.model_name + "fold " + str(i), train_start, train_end, test_score, file_path)
        else:
            save_results(parse.model_name, train_start, train_end, test_score, file_path)

        if args.divide_validata == True:
            model_predictions, true_labels = predict_confusion(model, test_dataset2, device=DEVICE)
        test2_score = estimate.evaluate(model_predictions, true_labels, args.threshold)
        if len(train_datasets) > 1:
            save_results(parse.model_name + "fold " + str(i), train_start, train_end, test2_score, file_path)
        else:
            save_results(parse.model_name, train_start, train_end, test2_score, file_path)

        # 打印评估结果
        print(f"{args.model_name}:{counter + 1}")
        print("测试集：")
        print_score(test_score)
        print_score(test2_score)
        df_test_score = pd.DataFrame(test_score, index=[0])
        if type(all_test_score) == int:
            all_test_score = df_test_score
        else:
            all_test_score = all_test_score + df_test_score

        break
        i = i + 1

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
        test_label = args.test_direction[:-8] + "_labels.npy"
        test_dataset = data_load_npy_predict(args.test_direction, test_label, args.batch_size)

    if args.Contrastive:
        model = TextCNN_confusion_Contrastive(args.filter_num, args.filter_size,
                                              args.output_size, args.dropout)
    elif args.confusion:
        model = TextCNN_confusion(args.filter_num, args.filter_size, args.output_size, args.dropout)
    else:
        model = TextCNN_CLS(args.filter_num, args.filter_size,
                            args.output_size, args.dropout, 768)
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
    # parse.model_name = 'tc_cbam_dro0.1'
    parse.model_num = 1
    parse.threshold = 0.5
    parse.epochs = 100
    parse.dropout = 0.6
    parse.confusion = True
    parse.model_name = "textCNN_example_val_100_0.6_5fc_dropoutBN"
    path = []
    parse.do_train = True
    parse.do_predict = False
    parse.divide_validata = True

    parse.Contrastive = False
    parse.early_stop = 10

    k_mer = 5
    crop_len = 225
    parse.dna_data_direction = os.path.join(parse.dna_data_direction, str(k_mer), str(crop_len))

    if parse.confusion:
        parse.model_name = parse.model_name + "_confusion"
    else:
        parse.model_name = parse.model_name + "_protein"

    if parse.Contrastive:
        parse.model_name = parse.model_name + "_Cont"

    for num in range(parse.model_num):
        a = f'saved_models/tc_cbam{num}.pth'
        path.append(a)

    for parse.k_fold in range(1):
        if parse.do_train:
            train(parse)
        if parse.do_predict:
            parse.model_name = parse.model_name + "_test"
            do_predict(parse)

    parse.filter_size_dna = [3, 4, 5, 8, 16, 32]
    parse.filter_size_protein = [3, 4, 5, 8, 16, 32]
    parse.filter_sizes = parse.filter_size_dna
    parse.model_name = "textCNN_example_val_100_0.6_5fc_dropoutBN_s_size_16_32"

    for parse.k_fold in range(1):
        if parse.do_train:
            train(parse)
        if parse.do_predict:
            parse.model_name = parse.model_name + "_test"
            do_predict(parse)

