# DPPred-indel
## Environment Preparation
Use the DPPred-indel environment.
```
conda activate DPPred-indel
```
## Data Preparation (under the dataCenter directory)
Training and testing sets:train.fasta  test.fasta 

## Execution
use scripts/extract.py cover fasta to npy
```
cd scripts
python extract.py
```
## Output Files
In the dataCenter directory, the corresponding feature and label npy files.


# DPPred-Cindel
## Environment Preparation
Use the DPPred-indel environment.
```
conda activate DPPred-indel
```
## Data Preparation (under the dataCenter directory)
(required)
Source domain training set: source_train.fasta

Source domain test set: test.fasta
(optional)
Source domain DA: source_Data.fasta

Target domain training and test sets: target_data.fasta, target_test.fasta

## Execution
use scripts/extract_C.py cover fasta to npy
```
cd scripts
python extract_C.py
```
## Output Files
In the dataCenter directory, the corresponding feature and label npy files.
