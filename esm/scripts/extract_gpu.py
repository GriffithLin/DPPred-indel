import torch
import esm
import numpy as np

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

# 载入 ESM-2 模型
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 禁用 dropout 以获取确定性的结果

if torch.cuda.is_available():
    model = model.cuda()
    print("模型已转移到 GPU")

dataset = FastaBatchedDataset.from_file("/data3/linming/DNA_Lin/esm/examples/data/test.fasta")
# batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=batch_converter, shuffle=False
)

# 定义保存文件的路径
# representations_file = 'data/train.npy'
# labels_file = 'data/train_labels.npy'

representations_file = 'data/test.npy'
labels_file = 'data/test_labels.npy'

# 打开文件以追加写入
with open(representations_file, 'ab') as repr_file, open(labels_file, 'ab') as labels_file:
    # 提取每个残基的表示（在 CPU 上）
    sequence_representations = []
    true_classes = []
    label_names = []
    lens = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            batch_lens = (toks != alphabet.padding_idx).sum(1)
            lens.append(batch_lens)
            print(batch_lens)
            # continue

            # print(
            #     f"处理 {batch_idx + 1} / {len(batches)} 批次（{toks.size(0)} 个序列）"
            # )
            if torch.cuda.is_available():
                print("to_cuda")
                toks = toks.to(device="cuda", non_blocking=True)
            results = model(toks, repr_layers=[33], return_contacts=False)

            token_representations = results["representations"][33]
            label_names.append(labels)
            # 获取每个序列的表示（不进行平均）
            for i, tokens_len in enumerate(batch_lens):
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

                sequence_representations.append(sequence_representation)
                true_class = float(labels[i].split("|")[-1])
                true_classes.append(true_class)

    # 将表示和标签转换为 NumPy 数组
    sequence_representations_np = np.array(sequence_representations)
    true_classes_np = np.array(true_classes)

    # 打印保存的文件的形状信息
    print(f"Saved Representations Shape: {sequence_representations_np.shape}")
    print(f"Saved Labels Shape: {true_classes_np.shape}")

    # 将表示和标签保存到 NumPy 文件
    np.save(repr_file, sequence_representations_np)
    np.save(labels_file, true_classes_np)

# # 如果你想保存标签名
# np.save('data/train_label_names.npy', np.array(label_names))
# np.save('data/train_lens.npy', np.array(lens))