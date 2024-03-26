import numpy as np
import pandas as pd
# # Memory-map the saved representations and labels
data_path = "/data3/linming/DNA_Lin/esm/scripts/data/"
# num_samples = 11313
# loaded_representations = np.memmap(data_path + 'train.npy', dtype=np.float32, mode='r', shape=(num_samples, 1024, 1280))
# loaded_labels = np.load(data_path + 'train_labels.npy')
# loaded_dna = np.load(data_path + 'train_dna.npy')
# # Check dimensions
# print("Loaded Representations Shape:", loaded_representations.shape)
# print("Loaded Labels Shape:", loaded_labels.shape)
# print("Loaded dna Shape:", loaded_dna.shape)


loaded_ddd_label = np.load(data_path + "target_test_labels.npy")
loaded_ddd_dna = np.load(data_path + "target_test_dna.npy")
loaded_ddd_protein = np.load(data_path + "target_test.npy")

print("Loaded loaded_ddd_dna Shape:", loaded_ddd_dna.shape)
print("Loaded loaded_ddd_protein Shape:", loaded_ddd_protein.shape)
print(loaded_ddd_label)