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
from my_util import get_config, data_load_npy_confusion_cont_k_fold, data_load_npy, data_load_npy_predict, \
    data_load_npy_confusion_predict, save_results, print_score, spent_time, data_load_confusion_k_fold
from models.model import TextCNN, TextCNN_confusion, TextCNN_CLS, TextCNN_confusion_Contrastive, ContrastiveLoss, \
    TextCNN_confusion_Embedding
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import estimate

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(DEVICE)

bases_kmer = 'ATCG'
bases = 'XATCG'

def train(args, paths=None):
    start_time = time.time()
    file_path = "{}/{}.csv".format('result', 'test')  # 结果保存路径

    print("Data is loading......（￣︶￣）↗　")
    # if args.k_fold != 1:

    if args.confusion == True:
        train_datasets, dev_datasets, test_dataset2 = data_load_confusion_k_fold(
            args.data_direction, args.dna_data_direction, args.batch_size)

    print("Data is loaded!�?≧▽�?)o")

    all_test_score = 0  # 初始话评估指�?
    # 训练并保存模�?
    counter = 1
    print(f"{args.model_name} is training......")
    i = 1
    for train_dataset, test_dataset in zip(train_datasets, dev_datasets):
        train_start = time.time()

        model = TextCNN_confusion_Embedding(args.filter_num, args.filter_size_dna, args.filter_size_protein,
                                  args.output_size, args.dropout)

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

        Train.train_step(train_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter,
                         early_stop=args.early_stop, threshold=args.threshold, printTestAcc=False)

        # 保存模型
        PATH = os.getcwd()
        each_model = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
        torch.save(model.state_dict(), each_model)

        # 模型预测
        if args.confusion == True:
            model_predictions, true_labels = predict_confusion(model, test_dataset, device=DEVICE)

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
        model = TextCNN_confusion(args.filter_num, args.filter_size_dna, args.filter_size_protein,
                                  args.output_size, args.dropout)
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


crop_len_list = [225]
k_mer_list = [5]

if __name__ == '__main__':
    parse = get_config()  # 获取参数

    # parse.model_name = 'tc_cbam_dro0.1'
    parse.model_num = 1
    parse.threshold = 0.5
    parse.epochs = 63
    parse.dropout = 0.7
    parse.confusion = True
    parse.model_name = "textCNN_val_63_0.7_5fc_dropout"
    path = []
    parse.do_train = True
    parse.do_predict = True
    parse.divide_validata = True

    parse.Contrastive = False
    parse.early_stop = 10

    if parse.confusion:
        parse.model_name = parse.model_name + "_confusion"
    else:
        parse.model_name = parse.model_name + "_protein"

    if parse.Contrastive:
        parse.model_name = parse.model_name + "_Cont"

    for num in range(parse.model_num):
        a = f'saved_models/tc_cbam{num}.pth'
        path.append(a)

    # 3, 4, 5, 8, 16, 32
    parse.filter_size_dna = [3, 4, 5, 8, 16, 32]
    parse.filter_size_protein = [3, 4, 5, 8, 16, 32]
    model_name = parse.model_name
    path = parse.dna_data_direction

    k_mer = 5
    crop_len = 225
    parse.dna_data_direction = os.path.join(path, str(k_mer), str(crop_len))
    parse.model_name = model_name + "_" + str(k_mer) + "_" + str(crop_len)
    print(parse)

    if parse.do_train:
        train(parse)
    if parse.do_predict:
        PATH = os.getcwd()
        each_model = os.path.join(PATH, 'saved_models', parse.model_name + '.pth')
        parse.model_path = each_model
        parse.model_name = parse.model_name + "_test"
        do_predict(parse)
