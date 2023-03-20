import torch
import torch.nn as nn


class toy_model(nn.Module):

    def __init__(self, num_classes) -> None:
        super(toy_model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        #self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, padding=2, padding_mode="reflect")
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.linear1 = nn.Linear(32*32*10, 1000)
        self.linear2 = nn.Linear(1000, 20)
        self.linear3 = nn.Linear(20, num_classes)
        self.activation = nn.ReLU()


    def forward(self, x):

        y = self.conv1(x)
        y = self.pooling(y)
        #y = self.conv2(y)
        #y = self.pooling(y)
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)

        return y
