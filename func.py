import numpy as np
import pandas as pd
import skimage.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_metrics(train_losses, train_accs, epochs):
    """
    画出loss和accuracy随epoch变化的曲线，并分别保存到文件。

    参数:
    train_losses: 训练集的loss列表
    train_accs: 训练集的accuracy列表
    epochs: epoch数列表
    """
    # 创建epochs列表
    if isinstance(epochs, int):
        epochs = list(range(1, epochs + 1))
    # 画出loss曲线
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.savefig('./results(CAF)/training_loss.png')
    plt.close()

    # 画出accuracy曲线
    plt.figure()
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Epochs')
    plt.legend()
    plt.savefig('./results(CAF)/training_accuracy.png')
    plt.close()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


# -------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch//2
    mirror_hsi = np.zeros(
        (height+2*padding, width+2*padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding+height),
               padding:(padding+width), :] = input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), i,
                   :] = input_normalize[:, padding-i-1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), width+padding+i,
                   :] = input_normalize[:, width-1-i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding*2-i-1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i, :,
                   :] = mirror_hsi[height+padding-1-i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(
        mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# -------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x+patch), y:(y+patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros(
        (x_train.shape[0], patch*patch*band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn*patch*patch:(nn+1)*patch*patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i*patch*patch:(i+1)*patch*patch,
                         :i+1] = x_train_reshape[:, :, band-i-1:]
            x_train_band[:, i*patch*patch:(i+1)*patch*patch,
                         i+1:] = x_train_reshape[:, :, :band-i-1]
        else:
            x_train_band[:, i:(i+1), :(nn-i)
                         ] = x_train_reshape[:, 0:1, (band-nn+i):]
            x_train_band[:, i:(i+1), (nn-i):] = x_train_reshape[:, 0:1, :(band-nn+i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch *
                         patch, :band-i-1] = x_train_reshape[:, :, i+1:]
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch *
                         patch, band-i-1:] = x_train_reshape[:, :, :i+1]
        else:
            x_train_band[:, (nn+1+i):(nn+2+i), (band-i-1):] = x_train_reshape[:, 0:1, :(i+1)]
            x_train_band[:, (nn+1+i):(nn+2+i), :(band-i-1)
                         ] = x_train_reshape[:, 0:1, (i+1):]
    return x_train_band


# -------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(
            mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(
            mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(
            mirror_image, true_point, k, patch)
    print("train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    # print("x_train_band shape = {}, type = {}".format(
    #     x_train_band.shape, x_train_band.dtype))
    # print("x_test_band  shape = {}, type = {}".format(
    #     x_test_band.shape, x_test_band.dtype))
    # print("x_true_band  shape = {}, type = {}".format(
    #     x_true_band.shape, x_true_band.dtype))
    # print("**************************************************")
    return x_train_band, x_test_band, x_true_band


# -------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    # print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    # print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    # print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    # print("**************************************************")
    return y_train, y_test, y_true


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


def csv2tiff(file_path):

    df = pd.read_csv(file_path, header=None, dtype=str, low_memory=False)

    # 从第二行，第三列开始读取数据
    data = df.iloc[1:, 2:-1].astype(float).values
    label = df.iloc[1:, -1].astype(int).values
    rows = df.iloc[1:, 0].astype(int).values.max() + 1
    cols = df.iloc[1:, 1].astype(int).values.max() + 1
    bands = data.shape[1]
    out_data = data.reshape(rows, cols, bands)
    out_label = label.reshape(rows, cols)

    sio.imsave('data/data.tif', out_data)
    sio.imsave('data/label.tif', out_label)
