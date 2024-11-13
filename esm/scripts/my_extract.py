import torch
import esm
import numpy as np
import pandas as pd

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 载入 ESM-2 模型
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 禁用 dropout 以获取确定性的结果

if torch.cuda.is_available():
    model = model.cuda()
    print("模型已转移到 GPU")

dataset = FastaBatchedDataset.from_file("/data3/linming/DNA_Lin/dataCenter/source_Data.fasta")
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=batch_converter, shuffle=False
)

# 定义保存文件的路径
representations_file = '/data3/linming/DNA_Lin/dataCenter/source_Data.npy'
labels_file = '/data3/linming/DNA_Lin/dataCenter/source_Data_labels.npy'

# 11313 1738 148
# 12819 5808 382
# 11114
num_samples = 12819
data_memmap = np.memmap(representations_file, dtype=np.float32, mode='w+', shape=(num_samples, 1024, 1280))
sequence_representations_list = []
true_classes = []
label_names = []
lens = []
j = 0
df_list = []
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        batch_lens = (toks != alphabet.padding_idx).sum(1)
        lens.append(batch_lens)
        print(batch_lens)

        if torch.cuda.is_available():
            print("to_cuda")
            toks = toks.to(device="cuda", non_blocking=True)
        results = model(toks, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]
        label_names.append(labels)
        # 获取每个序列的表示（不进行平均）
        for i, tokens_len in enumerate(batch_lens):
            tmp_dict = {}
            # 确保 tokens_len 在实际序列长度内
            tokens_len = min(tokens_len, 1024)

            # 提取整个序列的表示（包括填充部分）
            sequence_representation = token_representations[i, :tokens_len].cpu()

            # 如果序列长度小于最大长度，则填充表示
            if tokens_len < 1024:
                padding_length = 1024 - tokens_len
                padding = torch.zeros(padding_length, sequence_representation.size(-1))
                sequence_representation = torch.cat([sequence_representation, padding], dim=0)

            # 打印形状信息
            print(f"sequence_representation shape: {sequence_representation.shape}")
            sequence_representation = torch.unsqueeze(sequence_representation, dim = 0)
            data_memmap[j, :, :] = sequence_representation
            j += 1
            sequence_representations_list.append(sequence_representation)
            true_class = float(labels[i].split("|")[-1])
            variName = labels[i].split("|")[0]
            true_classes.append(true_class)
            tmp_dict["class"] = true_class
            tmp_dict["name"] = variName
            tmp_dict["strlen"] = int(tokens_len)
            df_list.append(tmp_dict)


    data_memmap.flush()
    del data_memmap
    true_classes_np = np.array(true_classes)

    # 打印保存的文件的形状信息
    print(f"Saved Labels Shape: {true_classes_np.shape}")

    np.save(labels_file, true_classes_np)
    print(j)

df_data = pd.DataFrame(df_list)
df_data.to_csv("/data3/linming/DNA_Lin/dataCenter/source_Data_list.csv")
# # 如果你想保存标签名
# np.save('data/train_label_names.npy', np.array(label_names))
# np.save('data/train_lens.npy', np.array(lens))