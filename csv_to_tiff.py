import pandas as pd
import numpy as np
import skimage.io as sio

# 读取CSV文件
file_path = 'D:/Python_Projects/8TRANS/results/S-TRANS/test/features_20s.csv' # 请将此处替换为您的CSV文件路径
df = pd.read_csv(file_path, header=None, dtype=str, low_memory=False)
print(df.shape)#(1740801, 131)
# 选择要查看的行数（例如第5行）
row_index = 4  # 第5行（Python中索引从0开始）

# 获取最后一列的列名
last_col = df.columns[-1]
# 打印原始数据中第row_index行的最后一列数据
print("原始数据中第{}行的最后一列数据：{}".format(row_index + 1, df.loc[row_index, last_col]))
# 将最后一列的数据不减1，改为不减
df[last_col] = pd.to_numeric(df[last_col], errors='coerce')
# 打印处理后的数据中第row_index行的最后一列数据
print("处理后的数据中第{}行的最后一列数据：{}".format(row_index + 1, df.loc[row_index, last_col]))

# 从第二行，第三列开始读取数据
data = df.iloc[1:, 2:-1].astype(float).values
label = df.iloc[1:, -1].astype(float).values

out_data = data.reshape(160, 3200, 128)#增强样本的大小应该是(160, 10880, 128)，测试20个样本的大小应该是(160, 3200, 128)
out_label = label.reshape(160, 3200)#增强样本的大小应该是(160, 10880)，测试20个样本的大小应该是(160, 3200)
print(out_data.shape)
print(out_label.shape)
sio.imsave('D:/Python_Projects/8TRANS/results/S-TRANS/test/features_20s.tiff', out_data)
sio.imsave('D:/Python_Projects/8TRANS/results/S-TRANS/test/label_20s.tiff', out_label)
