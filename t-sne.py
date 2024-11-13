from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

import cv2
from PIL import Image

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('device', device)

import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from train import predict, CosineScheduler, DataTrain_confusion, predict_confusion, predict_confusion_DA
from my_util import get_config, data_load_npy_confusion_cont, data_load_npy, save_results, print_score, spent_time, \
    data_load_npy_confusion_predict, data_load_npy_predict, data_load_npy_confusion_predict_DDD, \
    data_load_confusion_predict, data_load_confusion_predict_DDD
from models.model import TextCNN, TextCNN_CLS, TextCNN_confusion, TextCNN_confusion_noFan, \
    TextCNN_confusion_Contrastive, ContrastiveLoss, TextCNN_noFan, TextCNN_confusion_DA
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式


def do_predict(args):
    file_path = "{}/{}.csv".format('result', 'test')  # 评价指标保存

    print("data loading....")

    test_dataset = data_load_npy_confusion_predict(args.data_direction, args.dna_data_direction, args.batch_size)


    model = TextCNN_confusion(args.filter_num, args.filter_size_dna, args.filter_size_protein,
                                  args.output_size, args.dropout)

    print(model)

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    # 模型预测
    model = model.eval().to(device)
    model_trunc = create_feature_extractor(model, return_nodes={'FNN': 'semantic_feature'})
    encoding_array = []

    y_np = np.array([])
    pred_label_np = np.array([])
    with torch.no_grad():
        for x_dna, x_protein, y in test_dataset:
            x_dna = x_dna.to(device)
            x_protein = x_protein.to(device)
            y = y.to(device).unsqueeze(1)
            feature = model_trunc(x_dna, x_protein)[
                'semantic_feature'].squeeze().detach().cpu().numpy()  # 执行前向预测，得到 avgpool 层输出的语义特征
            score = model(x_dna, x_protein)
            pred_label = torch.sigmoid(score)
            encoding_array.extend(feature.tolist())
            # print(pred_label.shape)
            # print(encoding_array.shape)
            y_np = np.concatenate((y_np, y.squeeze().cpu().numpy()))
            pred_label_np = np.concatenate((pred_label_np, pred_label.squeeze().cpu().numpy()))
    results_df = np.stack((y_np, pred_label_np), axis=0)
    encoding_array = np.array(encoding_array)
    np.save('测试集语义特征.npy', encoding_array)
    np.save('测试集预测标签.npy', results_df)
    # results_df.to_csv('测试集预测标签.csv')

if __name__ == '__main__':
    parse = get_config()  # 获取参数

    parse.threshold = 0.5
    parse.confusion = True
    parse.batch_size = 64
    parse.epochs = 60
    parse.filter_size = [3, 4, 5, 8, 16, 32]
    parse.filter_size_dna = [3, 4, 5, 8, 16, 32]
    parse.filter_size_protein = [3, 4, 5, 8, 16, 32]
    parse.model_name = "textCNN_woKD_0.7_7_dropout0.6_BS64"
    path = []
    parse.do_train = True
    parse.do_predict = True

    parse.Contrastive = False
    parse.early_stop = 10

    k_mer = 5
    crop_len = 225
    parse.dna_data_direction = os.path.join(parse.dna_data_direction , str(k_mer), str(crop_len))




    print(parse)


    if parse.do_predict:
        PATH = os.getcwd()
        parse.model_name = "DPPred-indel"
        each_model = os.path.join(PATH, 'saved_models', parse.model_name + '.pth')
        parse.model_path = each_model
        parse.model_name = parse.model_name + "_test"
        do_predict(parse)

