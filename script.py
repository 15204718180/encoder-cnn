import os
import torch
import spectral
from config import *
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
import skimage.io as sio
import pandas as pd
from split_data import split_tran_and_test_labels
from func import *

device = set_device if torch.cuda.is_available() else 'cpu'


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        _, batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        _, batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        _, batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
# 输出评价指标
def output_metric(tar, pre, flag_test='train'):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    if flag_test == 'test':
        classification_result = classification_report(
            tar, pre, digits=4, zero_division=0)
        print('classification_result:\n', classification_result)
        print('confusion_matrix:\n', matrix)
        # 将结果写入文件
        with open('./results(CAF)/test_result.txt', 'w') as file:
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
# Parameter Setting
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
cudnn.deterministic = True
cudnn.benchmark = False

# 读取文件
file_extension = os.path.splitext(data_path)[1]
if file_extension == '.csv':
    csv2tiff(data_path)  # 调用一个函数将 CSV 文件转换为 TIFF 文件
    input = sio.imread(data_path)  # 读取转换后的 TIFF 文件
if file_extension == '.tif':
    input = sio.imread(data_path)  # 读取 TIFF 文件
if file_extension == '.npy':
        input = np.load(data_path) # 读取 NPY 文件
if file_extension == '.hdr':
        # 读取 ENVI 数据
        img = spectral.open_image(data_path)
        # 将数据转换为 numpy 数组
        input = img.load()
        input = np.array(input)
        print(input.shape)
if  file_extension == '.pt':
        input = torch.load(data_path)
        # 将数据转换为 numpy 数组
        input = np.array(input)
        print(input.shape)

# 分割数据集
if split_flag:
    split_tran_and_test_labels(label_path, train_ratio)
TR = sio.imread(train_label_path)
TE = sio.imread(test_label_path)

label = TR + TE
num_classes = np.max(TR)

# normalize data by band norm
input_normalize = np.zeros(input.shape)
# 进行归一化处理
for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    denominator = input_max - input_min
    if denominator == 0:
        # 如果分母为零，可以选择不进行归一化或使用默认值
        input_normalize[:, :, i] = 0  # 或者其他处理方法
    else:
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / denominator
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(
    TR, TE, label, num_classes)
mirror_image = mirror_hsi(
    height, width, band, input_normalize, patch=1)
x_train_band, x_test_band, x_true_band = train_and_test_data(
    mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=1, band_patch=band_patches)
y_train, y_test, y_true = train_and_test_label(
    number_train, number_test, number_true, num_classes)


# -------------------------------------------------------------------------------
# load data
x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(
    torch.FloatTensor)  # [695, 200, 7, 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(x_train, y_train)
x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(
    torch.FloatTensor)  # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset(x_test, y_test)
x_true = torch.from_numpy(x_true_band.transpose(
    0, 2, 1)).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(
    Label_train, batch_size=batch_size, shuffle=True)
label_test_loader = Data.DataLoader(
    Label_test, batch_size=batch_size, shuffle=True)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)


# -------------------------------------------------------------------------------
# create model
model = ViT(
    image_size=1,
    near_band=band_patches,
    num_patches=band,
    num_classes=num_classes,
    dim=64,
    depth=net_depth,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
    mode=my_mode,
    out_dims=output_dims
)
# 打开一个文件并将模型结构写入文件
with open('./results(CAF)/model_structure.txt', 'w') as f:
    f.write(str(model))
model = model.to(device)
# criterion
criterion = nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=epoches//10, gamma=0.9)


# -------------------------------------------------------------------------------
def test():
    model.load_state_dict(torch.load('./results(CAF)/SpectralFormer.pt'))
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v, flag_test='test')

    #混淆矩阵
    print(tar_v.shape)#(102405,)
    print(pre_v.shape)#(102405,)
    matrix = confusion_matrix(tar_v, pre_v)
    print('confusion_matrix:\n', matrix)

    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=np.uint8)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0],
                          total_pos_true[i, 1]] = pre_u[i]
    output_map = label_to_rgb(prediction_matrix)
    sio.imsave('./results(CAF)/predict_map.png', output_map)
    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(
        OA2, AA_mean2, Kappa2))
    # print(AA2)
    print("**************************************************")


def train():
    print("start training")
    tic = time.time()
    train_losses = []
    train_accs = []
    for epoch in range(epoches):
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(
            model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch+1, train_obj, train_acc))
        train_losses.append(train_obj.cpu().numpy())
        train_accs.append(train_acc.cpu().numpy())
        scheduler.step()
        if (epoch % 5 == 0) | (epoch == epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(
                model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")
    torch.save(model.state_dict(), './results(CAF)/SpectralFormer.pt')
    plot_metrics(train_losses, train_accs, epoches)

    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(
        OA2, AA_mean2, Kappa2))
    # print(AA2)
    print("**************************************************")


def encode_data():
    # 准备存储特征向量和标签的列表
    features_list = []
    labels_list = []
    # 加载模型, 设置模型为评估模式
    model.load_state_dict(torch.load('./results(CAF)/SpectralFormer.pt'))
    # 通过指定map_location参数将张量映射到CPU
    #model = torch.load('./results/S-TRANS/SpectralFormer_s.pt', map_location=torch.device('cpu'))
    # 创建模型实例
    #model = ViT()
    # 加载状态字典
    #state_dict = torch.load('./results/S-TRANS/SpectralFormer_s.pt', map_location=torch.device('cpu'))
    # 将状态字典加载到模型中
    #model.load_state_dict(state_dict)

    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        for (batch_data, batch_target) in label_true_loader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            feature, _ = model(batch_data)
            features_list.extend(feature.cpu().numpy())
            labels_list.extend(batch_target.cpu().numpy())

    all_features = np.zeros((height, width, band))
    all_labels = np.zeros((height, width))
    for i in range(total_pos_true.shape[0]):
        all_features[total_pos_true[i, 0],
                          total_pos_true[i, 1], :] = features_list[i]
        all_labels[total_pos_true[i, 0],
                          total_pos_true[i, 1]] = labels_list[i] - 1
    all_features = all_features.reshape(-1, band)
    all_labels = all_labels.reshape(-1)

    # 创建一个DataFrame
    df = pd.DataFrame(all_features)

    # 生成列名
    feature_names = ['Feature{}'.format(i+1) for i in range(df.shape[1])]
    df.columns = feature_names  # 更新特征列的列名

    # 添加标签列
    df['Label'] = all_labels

    # 添加行列序号
    num_rows = df.shape[0]  # 假设有这么多行数据
    rows = np.arange(num_rows) // input.shape[1]
    cols = np.arange(num_rows) % input.shape[1]

    df.insert(0, 'Column', cols)
    df.insert(0, 'Row', rows)

    # 将DataFrame保存到CSV文件
    df.to_csv('./results(CAF)/features_s.csv', index=False)
    print("特征向量已保存到features_s.csv文件中")
