import torch
import torch.nn as nn
from torchsummary import summary


class Conv3DNet(nn.Module):
    def __init__(self, bands=128, num_classes=10):
        super(Conv3DNet, self).__init__()
        self.features = nn.Sequential(#输入torch.Size([32, 1, 128, 9, 9])
            nn.Conv3d(1, 64, kernel_size=(7, 3, 3), stride=(2, 1, 1)),
            nn.ReLU(),#torch.Size([32, 64, 61, 7, 7])
            nn.Conv3d(64, 128, kernel_size=(5, 3, 3), stride=(2, 1, 1)),
            nn.ReLU(),#torch.Size([32, 128, 29, 5, 5])
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.ReLU(),#输出torch.Size([32, 256, 27, 3, 3])
        )
        self.conv2d1 = nn.Conv2d(6912, 1024, kernel_size=(1, 1))
        #原为self.conv2d1 = nn.Conv2d(6912, 1024, kernel_size=(1, 1))适用于128维度
        #nn.Conv2d(29440, 1024, kernel_size=(1, 1))适用于480维度
        self.conv2d2 = nn.Conv2d(1024, 512, kernel_size=(3, 3))
        self.output = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.unsqueeze(1)
        #print(x.shape)#torch.Size([32, 1, 128, 9, 9])
        x = self.features(x)
        #print(x.shape)#torch.Size([32, 256, 27, 3, 3])
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        #print(x.shape)#torch.Size([32, 6912, 3, 3])
        x = self.relu(self.conv2d1(x))
        #print(x.shape)#torch.Size([32, 1024, 3, 3])
        x = self.relu(self.conv2d2(x))
        #print(x.shape)#torch.Size([32, 512, 1, 1])
        x = self.flatten(x)
        #print(x.shape)#torch.Size([32, 512])
        x = self.output(x)
        #print(x.shape)#torch.Size([32, 10])
        return x

class HybridCNN(nn.Module):#加了降维
    def __init__(self, bands=480, num_classes=10):
        super(HybridCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.ReLU(),
        )
        self.conv2d = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.output = nn.Linear(576, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.relu(self.conv2d(x))
        x = self.flatten(x)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    window_size = 9
    bands = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Conv3DNet(num_classes=10).to(device)
    print(model)
    print(summary(model, (bands, window_size, window_size)))
