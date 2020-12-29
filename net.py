import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable


class Lankmark_net(nn.Module):
    def __init__(self):
        super(Lankmark_net, self).__init__()
        self.fc1 = nn.Linear(136, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def test():
    net = Lankmark_net()
    x = torch.randn((100, 136))
    a = net(x)
    print(x.size())
    print(a.size())


if __name__ == "__main__":
    test()
