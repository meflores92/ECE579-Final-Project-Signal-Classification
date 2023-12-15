import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_V2(nn.Module):
    def __init__(self):
        super(UNet_V2, self).__init__()

        # Contracting Path (Encoder)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive Path (Decoder)
        self.up_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.up_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        # Output layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)  # 3 output channels for 3 classes

    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.bn1(self.enc_conv1(x)), negative_slope=0.01)
        e2 = F.leaky_relu(self.bn2(self.enc_conv2(self.pool(e1))), negative_slope=0.01)
        e3 = F.leaky_relu(self.bn3(self.enc_conv3(self.pool(e2))), negative_slope=0.01)
        e4 = F.leaky_relu(self.bn4(self.enc_conv4(self.pool(e3))), negative_slope=0.01)

        # Decoder
        d1 = F.leaky_relu(self.bn5(self.dec_conv1(torch.cat([self.up_conv1(e4), e3], dim=1))), negative_slope=0.01)
        d2 = F.leaky_relu(self.bn6(self.dec_conv2(torch.cat([self.up_conv2(d1), e2], dim=1))), negative_slope=0.01)
        d3 = F.leaky_relu(self.bn7(self.dec_conv3(torch.cat([self.up_conv3(d2), e1], dim=1))), negative_slope=0.01)

        # Output
        out = self.out_conv(d3)
        return out
