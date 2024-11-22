import os
from config import *
from script import train, test, encode_data#

if __name__ == '__main__':

    # 训练模型
    train()

    # 测试模型
    test()

    # 编码数据
    #encode_data()
