#!/bin/bash

# 准备蛋白质特征
cd ./esm/scripts
#python my_extract.py

# 准备DNA特征
cd ../../DNABERT/examples
python exert_feature.py \
--model_type dna \
--tokenizer_name=dna5 \
--model_name_or_path \
/data3/linming/DNABERT/examples/embeding_model/5-new-12w-0/ \
--task_name dnaprom \
--do_predict \
--data_dir /data3/linming/DNA_Lin/dataCenter/5/100/ \
--max_seq_length 200 \
--per_gpu_pred_batch_size=128 \
--output_dir /data3/linming/DNABERT/examples/embeding_model/5-new-12w-0/ \
--predict_dir /data3/linming/DNA_Lin/esm/scripts/data/ \
--n_process 48

#运行程序
cd ../..
python DPPred-indel.py