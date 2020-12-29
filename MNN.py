import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()

#multiple channel mult
class Mult(nn.Module):
    def __init__(self, channels, size, times):
        super(Mult, self).__init__()
        self.channels = channels
        self.size = size
        self.times = times
        self.weights = []
        self.bias = []
        for i in range(times):
            self.weights.append(nn.Parameter(torch.Tensor(channels,size,size)))
            self.bias.append(nn.Parameter(torch.Tensor(channels,size,size)))
            torch.nn.init.normal_(self.weights[i], mean=0.0, std=0.1)

    def forward(self, input):
        feature_maps = []
        for i in range(self.times):
            temp = input * self.weights[i] + self.bias[i]
            feature_maps.append(temp)
        res = self.to_general(feature_maps)
        return res
    def to_general(self, feature_maps):
        result = feature_maps[0]
        for i in range(1, len(feature_maps)):
            result = torch.cat([result, feature_maps[i]], 1)
        return result


class MNN(nn.Module):
    def __init__(self):
        super(MNN,self).__init__()
        self.mult1 = Mult(1,32,40)
        self.mult2 = Mult(40,32,1)
        self.mult3 = Mult(40,32,1)

        self.conv = nn.Conv2d(40,40,3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9000, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.reshape(-1,1,32,32)
        in_size = x.size(0)
       # print(in_size)
        x = self.mult1(x)
        x = self.mult2(x)
        x = self.mult3(x)

       # print(x.size())
        x = self.conv(x)
        x = self.pool(x)
        #print(x.size())
        x = x.reshape(in_size,-1)
        #print(x.size())
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def test():
    net1 = Mult(1,32,40)
    net1 = net1.cuda()
    x = torch.randn((100,1,32,32))
    a = net1(x)
    print(a.size())



if __name__ == "__main__":
    test()
