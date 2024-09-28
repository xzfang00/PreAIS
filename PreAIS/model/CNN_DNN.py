import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class model_DNN_CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(model_DNN_CNN, self).__init__()
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5)
        self.dropout = nn.Dropout(p=0.2)
        # 全连接层
        self.fc1 = nn.Linear(180, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        # 卷积层
        out = self.conv1d(x)
        out = torch.relu(out)
        # 展平并应用 Dropout
        out = out.view(out.size(0), -1)  # 展平
        out = self.dropout(out)  # Dropout 层
        # 全连接层
        out = self.fc1(out)
        out = self.relu(out)
        # 再次应用 Dropout
        out = self.dropout(out)  # Dropout 层
        out = self.fc2(out)
        # 输出
        return self.sigmoid(out)