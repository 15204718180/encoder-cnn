import os
import glob
import torch
import spectral
import numpy as np
from tqdm import tqdm
import skimage.io as sio
import matplotlib.pyplot as plt
from model_3dcnn import Conv3DNet
from sklearn.model_selection import train_test_split
import pandas as pd

# 数据处理
# 在矩阵周围补0，使得每个像素点周围都有windowSize个像素点
def padWithZeros(X, margin=2):
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r +
                                margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1

    return patchesData, patchesLabels

class HsiDataset(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.tensor(Xtrain, dtype=torch.float32)
        self.y_data = torch.tensor(ytrain, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

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


def stack_data(data1, data2):
    if data1.shape[1] != data2.shape[1] or data1.shape[2] != data2.shape[2]:
        print("Error: The dimensions of the two datasets do not match.")
        exit()
    stack_data = np.concatenate((data1, data2), axis=0)
    return stack_data


def stack_label(data1, data2):
    if data1.shape[1] != data2.shape[1]:
        print("Error: The dimensions of the two datasets do not match.")
        exit()
    stack_data = np.concatenate((data1, data2), axis=0)
    return stack_data


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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, data_index, window_size):
        self.len = len(data_index)
        self.all_data = data
        self.data_index = data_index
        self.all_label = label
        self.window_size = window_size

    def __getitem__(self, index):
        cube_index = self.data_index[index]
        data_cube = get_image_cube(self.all_data, self.window_size, cube_index)
        data_cube = data_cube.transpose(2, 0, 1)
        label_index = self.all_label[index]
        # 将数据和标签转换为 tensor
        data_cube_tensor = torch.tensor(data_cube, dtype=torch.float32)
        label_index = int(label_index)
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        return data_cube_tensor, label_tensor

    def __len__(self):
        return self.len


class HsiDataSet(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置cube尺寸
    window_size = 9
    margin = int((window_size - 1) / 2)

    # 读取数据
    data_path = r"D:/Python_Projects/8TRANS/results/S-TRANS/features_s.tiff"
    label_path = r"D:/Python_Projects/8TRANS/results/S-TRANS/label_s.tiff"
    #data, label = load_data(data_path)
    # 读取融合特征 TIFF 文件
    data = sio.imread(data_path)
    # 读取标签 TIFF 文件
    label = sio.imread(label_path)
    # 数据归一化到0-1
    data = normalize(data)
    print(data.shape)  # (160, 10880, 128)
    data = padWithZeros(data, margin)

    print(data.shape)#(160, 10880, 128)#加入了补充的O为了分块
    print(label.shape)#(160, 10880)

    all_label = label.flatten()
    all_index = list(range(len(all_label)))

    # 将数据集按照3:7的比例分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(all_index, all_label, test_size=0.7, random_state=42)

    # 创建dataloader
    data_set = CustomDataset(data, y_train, X_train, window_size)
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=32, shuffle=True)
    test_set = CustomDataset(data, y_test, X_test, window_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False)

    # 创建模型，优化器等
    model = Conv3DNet(bands=128, num_classes=10)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 开始训练
    train_loss = []
    train_acc = []
    epoches = 10
    for epoch in range(epoches):
        # 训练模型
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        tqdm_loader = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epoches}")
        for i, (cube, label) in enumerate(tqdm_loader):
            cube, label = cube.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(cube)
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # 更新进度条
            running_loss += loss.item()
            tqdm_loader.set_postfix(loss=running_loss/(i+1))
        epoch_accuracy = correct_predictions / len(data_loader.dataset) * 100
        epoch_loss = running_loss/len(data_loader)
        train_loss.append(epoch_loss)
        print(
            f"Epoch {epoch+1}/{epoches}, Loss: {epoch_loss:.2f}, Acc: {epoch_accuracy:.2f}")

        # 验证模型
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        test_tqdm_loader = tqdm(test_loader, desc="val")
        for i, (cube, label) in enumerate(test_tqdm_loader):
            cube, label = cube.to(device), label.to(device)
            output = model(cube)
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == label).sum().item()
            loss = criterion(output, label)
            # 更新进度条
            running_loss += loss.item()
            test_tqdm_loader.set_postfix(loss=running_loss/(i+1))
        test_accuracy = correct_predictions / len(test_loader.dataset) * 100
        train_acc.append(test_accuracy)
        print(
            f"Test, Loss: {running_loss/len(test_loader):.2f}, Acc: {test_accuracy:.2f}")

    # 绘制loss和accuracy曲线
    plt.figure()
    plt.plot(range(1, epoches+1), train_loss, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/S-3DCNN/train_loss_s.png')#
    plt.close()

    plt.figure()
    plt.plot(range(1, epoches+1), train_acc, label='test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/S-3DCNN/train_acc_s.png')#
    plt.close()
    torch.save(model.state_dict(), "results/S-3DCNN/model_s.pth")

    # 创建一个包含train_loss和train_acc的DataFrame
    dataFrame = {'Epoch': range(1, len(train_loss) + 1), 'Loss': train_loss, 'Accuracy': train_acc}
    df = pd.DataFrame(dataFrame)

    # 将DataFrame保存为CSV文件
    df.to_csv('results/S-3DCNN/train_metrics_s.csv', index=False)#