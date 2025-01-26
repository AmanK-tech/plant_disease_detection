
import torch
from torch import nn

def conv(ni, nf, ks, stride=1, padding=1, act=True, bn=True, mp=True, dl=True):
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if act:
        layers.append(nn.ReLU())
    if mp:
        layers.append(nn.MaxPool2d(2, 2))
    if dl:
        layers.append(nn.Dropout2d(0.25))
    return nn.Sequential(*layers)

def lin(inp, op, dl=True, act=True):
    layers = [nn.Linear(inp, op)]
    if dl:
        layers.append(nn.Dropout(0.5))
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class CNNArchitecture(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.features = nn.Sequential(
            conv(3, 32, 3),
            conv(32, 64, 3),
            conv(64, 128, 3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            lin(128 * 16 * 16, 256),
            lin(256, 128),
            lin(128, num_classes, act=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn_arch = CNNArchitecture()




