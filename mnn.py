import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()

def weight_init(shape):
    weights = nn.Parameter(torch.randn(shape, requires_grad=True))
    return weights


def biases_init(shape):
    biases = nn.Parameter(torch.randn(shape, requires_grad=True))
    return biases


def to_general(feature_map):
    result = feature_map[0]
    for i in range(1, len(feature_map)):
        result = torch.cat([result, feature_map[i]], 1)
    return result


#multiple channel mult
class Mult1D(nn.Module):
    def __init__(self, channels, size, times):
        super(Mult1D, self).__init__()
        self.channels = channels
        self.size = size
        self.times = times
        self.weights = []
        self.bias = []
        for i in range(times):
            if use_cuda:
                self.weights.append(nn.Parameter(torch.Tensor(channels,size)).cuda())
                self.bias.append(nn.Parameter(torch.Tensor(channels,size)).cuda())
                torch.nn.init.normal_(self.weights[i], mean=0.0, std=0.1)
                torch.nn.init.normal_(self.bias[i], mean=0.0, std=0.1)
            else:
                self.weights.append(nn.Parameter(torch.Tensor(channels, size)))
                self.bias.append(nn.Parameter(torch.Tensor(channels, size)))
                torch.nn.init.normal_(self.weights[i], mean=0.0, std=0.1)
                torch.nn.init.normal_(self.bias[i], mean=0.0, std=0.1)

    def forward(self, input):
        feature_maps = []
        for i in range(self.times):
            temp = input * self.weights[i] + self.bias[i]
            # print('tensor_shape', input.shape, self.weights[i].shape, self.bias[i].shape, temp.shape)
            feature_maps.append(temp)
        res = self.to_general(feature_maps)
        return res

    def to_general(self, feature_maps):
        result = feature_maps[0]
        for i in range(1, len(feature_maps)):
            result = torch.cat([result, feature_maps[i]], 1)
        return result


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
            if use_cuda:
                self.weights.append(nn.Parameter(torch.Tensor(channels,size,size)).cuda())
                self.bias.append(nn.Parameter(torch.Tensor(channels,size,size)).cuda())
                torch.nn.init.normal_(self.weights[i], mean=0.0, std=0.1)
                torch.nn.init.normal_(self.bias[i], mean=0.0, std=0.1)
            else:
                self.weights.append(nn.Parameter(torch.Tensor(channels, size, size)))
                self.bias.append(nn.Parameter(torch.Tensor(channels, size, size)))
                torch.nn.init.normal_(self.weights[i], mean=0.0, std=0.1)
                torch.nn.init.normal_(self.bias[i], mean=0.0, std=0.1)

    def forward(self, input):
        feature_maps = []
        for i in range(self.times):
            #print(self.weights)
            temp = input * self.weights[i] + self.bias[i]
            feature_maps.append(temp)
        res = self.to_general(feature_maps)
        return res

    def to_general(self, feature_maps):
        result = feature_maps[0]
        for i in range(1, len(feature_maps)):
            result = torch.cat([result, feature_maps[i]], 1)
        return result


# weights list of [times][channels] similar performance much slower
class Mult2(nn.Module):
    def __init__(self, channels, size, times):
        super(Mult2, self).__init__()
        self.channels = channels
        self.size = size
        self.times = times
        self.weights = []
        self.bias = []

        for i in range(times):
            self.weights_time, self.bias_time = [], []
            for j in range(channels):
                # print(i,j)
                # print(len(self.weights_time))
                if use_cuda:
                    self.weights_time.append(nn.Parameter(torch.Tensor(1, size, size), requires_grad=True).cuda())
                    self.bias_time.append(nn.Parameter(torch.Tensor(1, size, size), requires_grad=True).cuda())
                    torch.nn.init.normal_(self.weights_time[j], mean=0.0, std=0.01)
                    torch.nn.init.normal_(self.bias_time[j], mean=0.0, std=0.01)
                else:
                    self.weights_time.append(nn.Parameter(torch.Tensor(1, size, size)))
                    self.bias_time.append(nn.Parameter(torch.Tensor(1, size, size)))
                    torch.nn.init.normal_(self.weights_time[j], mean=0.0, std=0.1)
                    torch.nn.init.normal_(self.bias_time[j], mean=0.0, std=0.01)
            self.weights.append(self.weights_time)
            self.bias.append(self.bias_time)

    def forward(self, input):
        # print(input.size())
        total_maps = []
        for i in range(self.times):
            feature_maps = []
            for j in range(self.channels):
                # print(self.channels)
                #print(j)
                #print('input resize', input[:, j, :, :].view(-1, 1, self.size, self.size).size())
                temp = input[:, j, :, :].view(-1, 1, self.size, self.size) * self.weights[i][j] + self.bias[i][j]
                #print('self.weights', self.weights[i][j].size())
                # print('temp',temp.size())
               # print('temp size', temp.size())
                feature_maps.append(temp)
                #print('len_featuremaps', len(feature_maps))
            res1 = self.to_general(feature_maps)
            total_maps.append(res1)
            #print('len total maps', len(total_maps))
        res = self.to_general(total_maps)
        #print('res size', res.size())
        return res

    def to_general(self, feature_maps):
        t = feature_maps[0]
        for i in range(1, len(feature_maps)):
            t = torch.cat([t, feature_maps[i]], 1)
        #print('typet', type(t))
        return t


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


class MNN1D(nn.Module):
    def __init__(self):
        super(MNN1D, self).__init__()
        self.mult1d1 = Mult1D(1, 1536, 20)
        self.mult1d2 = Mult1D(20, 1536, 1)
        self.mult1d3 = Mult1D(20, 1536, 1)
        self.fc1 = nn.Linear(30720, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = x.reshape(-1, 1, 1536)
        in_size = x.size(0)
        x = self.mult1d1(x)
        x = self.mult1d2(x)
        x = self.mult1d3(x)
        x = x.reshape(in_size, -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def test():
    net2 = Mult1D(1,1536,40).cuda()

    y = torch.randn(100,1,1536).cuda()
    b = net2(y)
    print(b.size())

    net3 = MNN1D().cuda()
    z = torch.randn(100, 1, 1536).cuda()
    c = net3(z)
    print(c.size())



if __name__ == "__main__":
    test()
