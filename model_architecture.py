
import torch
from torch import nn

def conv(ni, nf, ks, stride=1, padding=1, act=True, bn=True, mp=True, dl=False):

  layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)]
  if bn: layers.append(nn.BatchNorm2d(nf))
  if act: layers.append(nn.ReLU(inplace=True))  
  if mp: layers.append(nn.MaxPool2d(2, 2))
  return nn.Sequential(*layers)

class CNNArchitecture(nn.Module):
  def __init__(self, num_classes=15):
      super().__init__()
      self.features = nn.Sequential(
          conv(3, 32, 3),
          conv(32, 64, 3),
          conv(64, 128, 3),
          nn.AdaptiveAvgPool2d((4, 4))  
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(128 * 4 * 4, 256),
          nn.ReLU(inplace=True),
          nn.Linear(256, num_classes)
      )

  def forward(self, x):
      return self.classifier(self.features(x))

cnn_arch = CNNArchitecture()




