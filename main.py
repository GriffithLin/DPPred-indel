#!/usr/bin/env python
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
from train import DataTrain, predict, CosineScheduler
from my_util import get_config
from models.model import TextCNN, TextCNN_WithAttentionEncode, TextCNN_noEmbeding, TextCNN_WithAttentionEncode_noEmbeding
from models.tc_cbam import TC_CBAM
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


import estimate
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
print(DEVICE)

bases_kmer = 'ATCG'
bases = 'XATCG'


def mark_label(src):
    assert src.find("hgmd")>=0 or src.find("pos")>=0 or src.find("gnomAD")>=0 or src.find("neg")>=0 or src.find("Neg")>=0 or src.find("Pos")>=0

    if src.find("hgmd")>=0 or src.find("pos")>=0 or src.find("Pos")>=0:
        return 1
    return 0


def DivideValData(data):
    data:pd.DataFrame = data.sample(frac=1.0)                     #将数据打�?
    data.reset_index(drop=True)
    rows, cols = data.shape
    split_index_1 = int(rows * 0.2)
    #数据分割
    data_validate:pd.DataFrame = data.iloc[0: split_index_1, :]
    data_train:pd.DataFrame = data.iloc[split_index_1: rows, :]
    return data_train, data_validate


def getSequenceData(direction):
    # 从目标路径加载数�?
    data_Frame = pd.read_csv(direction)
    data = data_Frame["before_mutation"] + data_Frame["after_mutation"]
    label = data_Frame.apply(lambda x: mark_label(x["src"]), axis = 1)
    return np.array(data), np.array(label)

def getSequenceDataWithDivideValData(direction):
    # 从目标路径加载数�?
    data_Frame = pd.read_csv(direction)
    df_train, df_validate = DivideValData(data_Frame)

    data_train = df_train["before_mutation"] + df_train["after_mutation"]
    label_train = df_train.apply(lambda x: mark_label(x["src"]), axis=1)

    data_val = df_validate["before_mutation"] + df_validate["after_mutation"]
    label_val = df_validate.apply(lambda x: mark_label(x["src"]), axis = 1)

    return np.array(data_train), np.array(label_train), np.array(data_val), np.array(label_val)

def getSequenceData_Kmer(direction, k_mer):
    data_Frame = pd.read_csv(direction)

    if str(k_mer) + "mer_before" not in data_Frame.columns:
        data1 = []
    else:
        data1 = data_Frame[str(k_mer) + "mer_before"]

    data2 =data_Frame[str(k_mer) + "mer_after"]
    label = data_Frame.apply(lambda x: mark_label(x["src"]), axis=1)
    return np.array(data1),np.array(data2), np.array(label)

# TODO!!
def getSequenceData_KmerWithDivideValData(direction, k_mer):
    # 从目标路径加载数�?
    data_Frame = pd.read_csv(direction)
    df_train, df_validate = DivideValData(data_Frame)

    data_train = df_train["before_mutation"] + df_train["after_mutation"]
    label_train = df_train.apply(lambda x: mark_label(x["src"]), axis=1)

    data_val1 = df_validate[str(k_mer) + "mer_before"]
    data_val2 = df_validate[str(k_mer) + "mer_after"]
    label_val = df_validate.apply(lambda x: mark_label(x["src"]), axis=1)

    return np.array(data_train), np.array(label_train), np.array(data_val1), np.array(label_val)

def PadEncode(data, label, max_len=1022):
    # 序列编码
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        if len(data[i]) > max_len:  # 剔除序列长度大于max_len的序�?
            print( "biger than max_len" + str(len(data[i])))
            continue
        element, st = [], data[i].strip()
        for j in st:
            if j not in bases:  # 剔除包含非天然氨基酸的序�?
                sign = 1
                break
            index = bases.index(j)
            element.append(index)
            sign = 0

        if length <= max_len and sign == 0:  # 序列长度小于50且只包含天然氨基酸的序列
            temp.append(element)
            seq_length.append(len(temp[b]))  # 保存序列有效长度
            b += 1
            element += [0] * (max_len - length)  # �?补齐序列长度
            data_e.append(element)
            label_e.append(label[i])
    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e))



def code_kmer(token, k):
    result = 0
    for i in range(k):
        result *= len(bases_kmer)
        result += bases_kmer.index(token[i])
    return result + 1

def code_kmer_seq(kmer_seq, k):
    i = 0
    j = k
    coded = []
    while j <= len(kmer_seq):
        token = kmer_seq[i:j]
        coded.append(code_kmer(token, k))
        i += (k+1)
        j += (k+1)
    return np.array(coded)

#  不填充\定长,指序列长度等于最大长�?
def PadEncode_kmer(data1, data2, label, max_len=50, kmer=6):
    kmer_length = (max_len-kmer+1)* (kmer +1) - 1
    # df1 = pd.DataFrame(data1)
    # df2 = pd.DataFrame(data2)


    # 序列编码
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0

    if len(data1) == 0:
        for i in range(len(data2)):
            if  (len(data2[i]) != kmer_length):
                print(len(data2[i]))
                print(kmer_length)
                continue

            element2 = code_kmer_seq(data2[i], kmer)

            # element1 += [0] * (kmer_max - length1)  # �?补齐序列长度
            # element2 += [0] * (kmer_max - length2)

            data_e.append(element2)
            label_e.append(label[i])
    else:
        for i in range(len(data2)):
            if (len(data1[i]) != kmer_length) or (len(data2[i]) != kmer_length):
                print(len(data1[i]))
                print(len(data2[i]))
                print(kmer_length)
                continue

            element1 = code_kmer_seq(data1[i], kmer)
            element2 = code_kmer_seq(data2[i], kmer)

            # element1 += [0] * (kmer_max - length1)  # �?补齐序列长度
            # element2 += [0] * (kmer_max - length2)

            data_e.append(np.concatenate([element1, element2],axis=0))
            label_e.append(label[i])
    print(np.array(data_e).shape)
    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e))



class NpyDataset(Dataset):
    def __init__(self, npy_file, label_file):
        self.data = np.lib.format.open_memmap(npy_file)
        self.labels =np.lib.format.open_memmap(label_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])
        return (self.data[idx]), (self.labels[idx])


def data_load_npy(train_path, test_path, batch=32):
    train_label = train_path[:-4] + "_labels" + train_path[-4:]
    test_label = test_path[:-4] + "_labels" + test_path[-4:]
    datasets_train, datasets_test = [], []


    # x_train = np.lib.format.open_memmap(train_path)
    # y_train = np.lib.format.open_memmap(train_label)
    #
    # x_test = np.lib.format.open_memmap(test_path)
    # y_test = np.lib.format.open_memmap(test_label)
    #
    # x_train = torch.from_numpy(x_train).long()
    # x_test = torch.from_numpy(x_test).long()
    #
    # # Create datasets
    # dataset_train = TensorDataset(x_train, y_train)
    # dataset_test = TensorDataset(x_test, y_test)

    dataset_train = NpyDataset(train_path, train_label)
    dataset_test = NpyDataset(test_path, test_label)

    # dataset_val = TensorDataset(x_val, y_val)
    # dataset_val = DataLoader(dataset_val, batch_size=batch, shuffle=True)

    dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    datasets_train.append(dataset_train)
    datasets_test.append(dataset_test)
    # datasets_val.append(dataset_val)

    # return datasets_train, dataset_val, datasets_test
    return datasets_train, datasets_test

def data_load(train_direction, test_direction, dataset_direction, max_length, batch, k_mer, encode='embedding'):
    datasets_train, datasets_test = [], []
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'
    if ((train_direction is not None) or (test_direction is not None)) and (dataset_direction is None):
        # 从目标路径加载数�?
        if k_mer == 0:
                train_sequence_data, train_sequence_label = getSequenceData(train_direction)
                test_sequence_data, test_sequence_label = getSequenceData(test_direction)

                # 选择序列编码方式
                # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
                x_train, y_train= PadEncode(train_sequence_data, train_sequence_label, max_length)
                x_test, y_test= PadEncode(test_sequence_data, test_sequence_label, max_length)
        else:
            train_sequence_data_before, train_sequence_data_after, train_sequence_label = getSequenceData_Kmer(train_direction, k_mer)
            test_sequence_data_before, test_sequence_data_after, test_sequence_label = getSequenceData_Kmer(test_direction, k_mer)

            # 选择序列编码方式
            # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
            print("encode train")
            x_train, y_train= PadEncode_kmer(train_sequence_data_before, train_sequence_data_after, train_sequence_label, max_length, k_mer)
            print("encode test")
            x_test, y_test= PadEncode_kmer(test_sequence_data_before, test_sequence_data_after, test_sequence_label, max_length, k_mer)

        # Create datasets
        dataset_train = TensorDataset(x_train, y_train)
        dataset_test = TensorDataset(x_test, y_test)

        dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=True)

        datasets_train.append(dataset_train)
        datasets_test.append(dataset_test)

    elif dataset_direction is not None:
        train_sequence, train_label = getSequenceData(dataset_direction)
        encode_data, encode_label = PadEncode(train_sequence, train_label, max_length)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        for i, (train_index, test_index) in enumerate(cv.split(encode_data, encode_label)):
            data_train, label_train = encode_data[train_index], encode_label[train_index]
            data_test, label_test = encode_data[test_index], encode_label[test_index]
            train_data = TensorDataset(data_train, label_train)
            test_data = TensorDataset(data_test, label_test)
            datasets_train.append(DataLoader(train_data, batch_size= batch, shuffle=True))
            datasets_test.append(DataLoader(test_data, batch_size= batch, shuffle= True))

    return datasets_train, datasets_test

# TODO
def data_load_withVal(train_direction, test_direction, max_length, batch, k_mer, encode='embedding'):
    datasets_train, datasets_test = [], []
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'
    # 从目标路径加载数�?
    if k_mer == 0:
            train_sequence_data, train_sequence_label, val_sequence_data, val_sequence_label = getSequenceDataWithDivideValData(train_direction)
            test_sequence_data, test_sequence_label = getSequenceData(test_direction)

            # 选择序列编码方式
            # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
            x_train, y_train = PadEncode(train_sequence_data, train_sequence_label, max_length)
            x_val, y_val = PadEncode(val_sequence_data, val_sequence_label, max_length)
            x_test, y_test = PadEncode(test_sequence_data, test_sequence_label, max_length)
    else:
        print("TODO")
        assert False
        train_sequence_data_before, train_sequence_data_after, train_sequence_label = getSequenceData_Kmer(train_direction, k_mer)
        test_sequence_data_before, test_sequence_data_after, test_sequence_label = getSequenceData_Kmer(test_direction, k_mer)

        # 选择序列编码方式
        # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
        print("encode train")
        x_train, y_train= PadEncode_kmer(train_sequence_data_before, train_sequence_data_after, train_sequence_label, max_length, k_mer)
        print("encode test")
        x_test, y_test= PadEncode_kmer(test_sequence_data_before, test_sequence_data_after, test_sequence_label, max_length, k_mer)




    # Create datasets
    dataset_train = TensorDataset(x_train, y_train)
    dataset_test = TensorDataset(x_test, y_test)

    dataset_val = TensorDataset(x_val, y_val)
    dataset_val = DataLoader(dataset_val, batch_size=batch, shuffle=True)

    dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    datasets_train.append(dataset_train)
    datasets_test.append(dataset_test)
    # datasets_val.append(dataset_val)

    return datasets_train, dataset_val, datasets_test




def spent_time(start, end):
    epone = int(epoch_time / 60)  # 分钟
    secs = int(epoch_time - minute * 60)  # �?
    return minute, secs

# '%.3f' % test_score[5],
def save_results(model_name, start, end, test_score, file_path):
    # 保存模型结果 csv文件
    title = ['Model']
    title.extend(test_score.keys())
    title.extend(['RunTime', 'Test_Time'])

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content_row = [model_name]
    for key in test_score:
        content_row.append('%.3f' % test_score[key])
    content_row.extend([[end - start], now])

    content = [content_row]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None)
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:  # newline用来控制空的行数
                writer = csv.writer(t)  # 创建一个csv的写入器
                writer.writerows(content)  # 写入数据
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)  # 写入标题
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

def print_score(test_score):
    if type(test_score) == dict:
        for key in test_score:
            print(key + f': {test_score[key]:.3f}')
    else:
        print(test_score)

def main(args, paths=None):
    start_time = time.time()
    file_path = "{}/{}.csv".format('result', 'test')  # 结果保存路径

    print("Data is loading......（￣︶￣）↗　")
    if args.divide_validata:
        train_datasets, val_dataset, test_datasets = data_load_withVal(args.train_direction, args.test_direction, args.max_length,
                                                args.batch_size, args.k_mer, encode='embedding')  # 加载训练数据和测试数据，并编�?
    elif args.load_npy:
        train_datasets, test_datasets = data_load_npy(args.train_direction, args.test_direction, args.batch_size)
    else:
        train_datasets, test_datasets = data_load(args.train_direction, args.test_direction, args.dataset_direction, args.max_length,
                                                args.batch_size, args.k_mer, encode='embedding')  # 加载训练数据和测试数据，并编�?
    print("Data is loaded!�?≧▽�?)o")

    all_test_score = 0  # 初始话评估指�?
    # 训练并保存模�?
    if paths is None:
        print(f"{args.model_name} is training......")
        for i in range(len(train_datasets)):
            train_dataset = train_datasets[i]
            test_dataset = test_datasets[i]
            for counter in range(args.model_num):
                train_start = time.time()
                if args.load_npy:
                    # 初始化相关参�?
                    if args.attentionENcode:

                        model = TextCNN_WithAttentionEncode_noEmbeding(args.filter_num, args.filter_size, args.output_size, args.dropout, args.max_length)  # 初始化模�?
                    else:

                        model = TextCNN_noEmbeding(args.filter_num, args.filter_size,
                                                            args.output_size, args.dropout)  # 初始化模�?
                else:
                    # 初始化相关参�?
                    if args.attentionENcode:
                        model = TextCNN_WithAttentionEncode(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                                        args.output_size, args.dropout, args.max_length)  # 初始化模�?
                        # print(TextCNN_noEmbeding)
                        # model = TextCNN_noEmbeding(args.filter_num, args.filter_size, args.output_size, args.dropout)  # 初始化模�?
                    else:
                        model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                                                            args.output_size, args.dropout)  # 初始化模�?
                        # print(TextCNN_noEmbeding)
                        # model = TextCNN_noEmbeding(args.filter_num, args.filter_size,
                        #                                     args.output_size, args.dropout)  # 初始化模�?

                print(model)

                if args.opt == "SGD":
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)  # 优化�?
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 优化�?
                lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500
                                               )  # 退化学习率
                criterion = torch.nn.BCEWithLogitsLoss()  # 损失函数
                # 初始化训练类
                Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

                # 训练模型


                if args.divide_validata:
                    Train.train_step_val(train_dataset, val_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter, early_stop=args.early_stop, threshold = args.threshold)
                else:
                    Train.train_step(train_dataset, test_dataset, args.model_name, epochs=args.epochs, model_num=counter, early_stop=args.early_stop, threshold = args.threshold)

                # 保存模型
                PATH = os.getcwd()
                each_model = os.path.join(PATH, 'saved_models', args.model_name + str(counter) + '.pth')
                torch.save(model.state_dict(), each_model)

                # 模型预测
                model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)

                # 模型评估
                test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)

                # 保存评估结果
                train_end = time.time()
                if len(train_datasets)>1:
                    save_results(parse.model_name + "fold " + str(i), train_start, train_end, test_score, file_path)
                else:
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
    else:
        for i in range(len(train_datasets)):
            test_dataset = test_datasets[i]
            # 测试模型
            for model_path in paths:
                EStart_Time = time.time()
                # 初始化模�?
                model = TC_CBAM(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                                args.output_size, args.dropout)
                # 加载模型
                model.load_state_dict(torch.load(model_path))
                # 模型预测
                model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
                # 模型评估
                test_score = estimate.evaluate(model_predictions, true_labels, args.threshold)
                print(f"{args.model_name}:")
                print("测试集：")
                print_score(test_score)

                df_test_score = pd.DataFrame(test_score, index=[0])
                if type(all_test_score) == int:
                    all_test_score = df_test_score
                else:
                    all_test_score = all_test_score + df_test_score

                EEnd_Time = time.time()
                # 保存模型性能
                save_results(parse.model_name, EStart_Time, EEnd_Time, test_score, file_path)

    "-------------------------------------------打印平均结果-----------------------------------------------"
    run_time = time.time()
    m, s = spent_time(start_time, run_time)  # 运行时间
    print(f"runtime:{m}m{s}s")
    print("测试集：")
    all_test_score = all_test_score / (len(train_datasets) * args.model_num)
    print_score(all_test_score)
    save_results('average', start_time, run_time, all_test_score, file_path)
    "---------------------------------------------------------------------------------------------------"


if __name__ == '__main__':
    parse = get_config()  # 获取参数
    print(parse)
    # parse.model_name = 'tc_cbam_dro0.1'
    # parse.model_num = 1
    # parse.threshold = 0.5
    path = []
    for num in range(parse.model_num):
        a = f'saved_models/tc_cbam{num}.pth'
        path.append(a)
    main(parse)
