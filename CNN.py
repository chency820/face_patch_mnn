from torchvision import datasets, models, transforms
from torch import nn

resnet1 = models.resnet18(pretrained=True)
resnet1.fc = nn.Linear(512, 6)

print(resnet1)