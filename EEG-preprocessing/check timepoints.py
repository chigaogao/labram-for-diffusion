import numpy as np

loaded_data = np.load("D:\eeg-things数据集\Preprocessed_data_2000Hz\sub-01\preprocessed_eeg_training.npy",allow_pickle=True)
# --- 3. 查看加载后的数据 ---
print(loaded_data)