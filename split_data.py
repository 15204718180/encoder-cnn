# 该程序用于分割训练数据和测试数据
import os
from osgeo import gdal
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split


def split_tran_and_test_labels(label_tiff_path, train_ratio=0.05, seed=42):
    print(f'Training data ratio: {train_ratio}')
    # 使用 skimage.io 读取标签图
    labels = io.imread(label_tiff_path)
    labels = labels + 1#原本标签图像中的所有标签值都变大了1,背景像素（现在的值为1）会被跳过，确保只处理实际有意义的标签值
    # 创建训练和测试标签图，初始化为0（假设0是背景或无标签）
    train_labels = np.zeros_like(labels, dtype=np.uint8)
    test_labels = np.zeros_like(labels, dtype=np.uint8)

    # 处理每个类别
    for label in np.unique(labels):
        if label == 0:
            continue  # 跳过背景或无标签的部分
        # 找到当前类别的所有像素位置
        indices = np.where(labels == label)
        # 分割为训练和测试
        train_idx, test_idx = train_test_split(
            range(len(indices[0])), train_size=train_ratio, random_state=seed)
        # 设置训练和测试标签
        train_labels[indices[0][train_idx], indices[1][train_idx]] = label
        test_labels[indices[0][test_idx], indices[1][test_idx]] = label

    # 使用 skimage.io 保存图像
    io.imsave('D:/Python_Projects/9data-enhance/enhance_samples-2+14+20samples/enhance_samples_combined-2+14+20_t_train.tiff', train_labels)
    io.imsave('D:/Python_Projects/9data-enhance/enhance_samples-2+14+20samples/enhance_samples_combined-2+14+20_t_test.tiff', test_labels)

    print("训练和测试标签图已保存为 'enhance_samples_combined-2+14+20_t_train.tiff' 和 'enhance_samples_combined-2+14+20_t_test.tiff'。")


if __name__ == '__main__':
    split_tran_and_test_labels('D:/Python_Projects/9data-enhance/enhance_samples-2+14+20samples/enhance_samples_combined-2+14+20_t.tif', train_ratio=0.8)
