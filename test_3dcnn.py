import os
import glob
import torch
import spectral
import numpy as np
from tqdm import tqdm
import skimage.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
from model_3dcnn import Conv3DNet
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from train_3dcnn import CustomDataset
from sklearn.metrics import confusion_matrix

# 数据处理
# 在矩阵周围补0，使得每个像素点周围都有windowSize个像素点
def padWithZeros(X, margin=2):
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# -------------------------------------------------------------------------------
def get_image_cube(data, window_size, index):
    margin = int((window_size - 1) / 2)
    # img_rows = data.shape[0] - 2 * margin
    img_cols = data.shape[1] - 2 * margin
    data_cube = np.zeros((window_size, window_size, data.shape[2]))
    index_row = index // img_cols + margin
    index_col = index % img_cols + margin

    data_cube = data[index_row - margin:index_row + margin + 1,
                     index_col - margin:index_col + margin + 1, :]
    return data_cube

# -------------------------------------------------------------------------------
def stack_data(data1, data2):
    if data1.shape[1] != data2.shape[1] or data1.shape[2] != data2.shape[2]:
        print("Error: The dimensions of the two datasets do not match.")
        exit()
    stack_data = np.concatenate((data1, data2), axis=0)
    return stack_data

# -------------------------------------------------------------------------------
def stack_label(data1, data2):
    if data1.shape[1] != data2.shape[1]:
        print("Error: The dimensions of the two datasets do not match.")
        exit()
    stack_data = np.concatenate((data1, data2), axis=0)
    return stack_data

# -------------------------------------------------------------------------------
def load_data(data_path):
    # 构建 .hdr 文件的路径模式
    hdr_pattern = os.path.join(data_path, '*.hdr')
    # 构建 .tif 文件的路径模式
    tif_pattern = os.path.join(data_path, '*.tif')
    # 使用 glob 获取所有匹配的文件路径
    hdr_files = glob.glob(hdr_pattern)
    tif_files = glob.glob(tif_pattern)
    # 确保 .hdr 和 .tif 文件的数量相等
    if len(hdr_files) != len(tif_files):
        print("Error: The number of .hdr and .tif files do not match.")
        exit()
    all_data = []
    all_label = []
    for i in range(len(hdr_files)):
        data = spectral.open_image(hdr_files[i])
        data = data.load()
        data = np.array(data)
        label = sio.imread(tif_files[i])
        if i > 0:
            all_data = stack_data(all_data, data)
            all_label = stack_label(all_label, label)
        else:
            all_data = data
            all_label = label

    return all_data, all_label

# -------------------------------------------------------------------------------

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# -------------------------------------------------------------------------------
# 输出评价指标
def output_metric(tar, pre, flag_test='train'):
    matrix = confusion_matrix(tar, pre)
    # 可视化混淆矩阵并保存
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('results/S-3DCNN/test/confusion_matrix_20s.png'))
    plt.close()  # 关闭图像以避免显示
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    if flag_test == 'test':
        classification_result = classification_report(
            tar, pre, digits=4, zero_division=0)
        print('classification_result:\n', classification_result)
        print('confusion_matrix:\n', matrix)
        # 将结果写入文件
        with open('./results/S-3DCNN/test/test_result_20s.txt', 'w') as file:
            file.write('Classification Result:\n')
            file.write(classification_result + '\n')
            file.write('Confusion Matrix:\n')
            file.write(str(matrix) + '\n')
            file.write('Overall Accuracy (OA): {}\n'.format(OA))
            file.write('Average Accuracy (AA_mean): {}\n'.format(AA_mean))
            file.write('Kappa Coefficient: {}\n'.format(Kappa))
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + 1e-10)
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
# -------------------------------------------------------------------------------
# 将label图像转换为RGB图像的函数
def label_to_rgb(label_img):
    # 定义字典
    label_to_color = {
        0: (0, 0, 0),  # 黑色
        1: (255, 0, 255),  # 粉色
        2: (0, 100, 0),  # 深绿色
        3: (0, 0, 205),  # 深蓝色
        4: (255, 255, 0),  # 黄色
        5: (255, 127, 36),  # 橘色
        6: (0, 255, 255),  # 天蓝色
        7: (0, 255, 0),  # 浅绿色
        8: (255, 225, 255),  # 浅红色
        9: (255, 0, 0)  # 正红色
    }
    # 获取label图像的尺寸
    height, width = label_img.shape
    # 创建一个新的RGB图像
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 将label转换为RGB
    for label, color in label_to_color.items():
        rgb_img[label_img == label] = color

    return rgb_img

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = r"D:/Python_Projects/8TRANS/results/S-TRANS/test/features_20s.tiff"
    label_path = r"D:/Python_Projects/8TRANS/results/S-TRANS/test/label_20s.tiff"

    # 读取 ENVI 数据hdr格式
    data = sio.imread(data_path)
    # 读取标签 TIFF 文件
    label = sio.imread(label_path)
    print(data.shape)#(160, 3200, 128)
    print(label.shape)#(160, 3200)
    all_labels = label.flatten()
    all_index = list(range(len(all_labels)))

    data = normalize(data)
    bands = data.shape[2]
    window_size = 9
    margin = int((window_size - 1) / 2)
    data = padWithZeros(data, margin)

    model = Conv3DNet(bands=128, num_classes=10)
    model.load_state_dict(torch.load(
        'results/S-3DCNN/model_s.pth', map_location='cpu'))
    model.to(device)
    data_set = CustomDataset(data, all_labels, all_index, window_size)
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=128, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    correct_predictions = 0
    tqdm_loader = tqdm(data_loader, desc=f"eval")
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for i, (cube, label) in enumerate(tqdm_loader):
            cube, label = cube.to(device), label.to(device)
            output = model(cube)
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            loss = criterion(output, label)
            # 更新进度条
            running_loss += loss.item()
            tqdm_loader.set_postfix(loss=running_loss/(i+1))
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        epoch_accuracy = correct_predictions / len(data_loader.dataset) * 100
        print(
            f"Loss: {running_loss/len(data_loader):.2f}, Acc: {epoch_accuracy:.2f}")

    # 计算分类结果
    OA, AA_mean, Kappa, AA = output_metric(
        all_labels, all_predictions, flag_test='test')
    print(f"OA: {OA:.2f}, AA_mean: {AA_mean:.2f}, Kappa: {Kappa:.2f}")
    print(f"AA: {AA}")

    all_predictions = np.array(all_predictions)
    print(all_predictions.shape)
    rgb_img = label_to_rgb(all_predictions.reshape(160, 3200))
    # 显示RGB图像
    #rgb_img.show()
    sio.imsave('results/S-3DCNN/test/predict_map_20s.png', rgb_img)