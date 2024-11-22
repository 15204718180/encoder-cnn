# 配置文件
# 输出参数
split_flag = False  # 是否重新分割数据
data_path = 'D:/Python_Projects/8trans/combined 20 def/combined_20.hdr'     # 输入文件夹数据路径#combined_20_s+t+e.npy
train_label_path = 'D:/Python_Projects/8trans/combined 20 def/combined_20_label_train.tiff'   # 标签数据路径
test_label_path = 'D:/Python_Projects/8trans/combined 20 def/combined_20_label_test.tiff'   # 标签数据路径
net_depth = 5                   # 网络深度(encoder层数)
output_dims = 128               # 保存的向量维度
train_ratio = 0.8               # 训练集比例(已在标签中分好，这里并未使用这个参数# )

# 训练参数
epoches = 100                    # 训练轮数
set_device = 'cuda:0'           # GPU编号
learning_rate = 5e-4            # 学习率
my_seed = 1                     # 随机种子
band_patches = 1                # 分组编码
batch_size = 32                 # 批次大小
my_mode = 'ViT'                 # 网络结构CAF或ViT
