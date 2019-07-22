import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


class CNNModel(nn.Module):
    def __init__(self, nclasses=2):
        super(CNNModel, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.fc = nn.Sequential(nn.Linear(1000, nclasses, bias=False))

    def forward(self, x):
        x = self.fc(self.resnet(x))
        return x
 
