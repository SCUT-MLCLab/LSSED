# -*- coding: utf-8 -*-
# @Describe : vgg11 as Backbone
# @Time : 2022/03/26 21:49
# @Author : zpx
# @File : network.py
import math

import torch
import torch.nn as nn
from torchvision.models import vgg11
from vit import Vit


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = vgg11(pretrained=True).features
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = nn.Sequential(  # (b,128,224,224)
            self.encoder[0], self.encoder[1],
            self.encoder[3], self.encoder[4]
        )
        self.encoder2 = nn.Sequential(  # (b,256,112,112)
            self.encoder[6], self.encoder[7],
            self.encoder[8], self.encoder[9]
        )
        self.encoder3 = nn.Sequential(  # (b,512,56,56)
            self.encoder[11], self.encoder[12],
            self.encoder[13], self.encoder[14]
        )
        self.encoder4 = nn.Sequential(  # (b,512,28,28)
            self.encoder[16], self.encoder[17],
            self.encoder[18], self.encoder[19]
        )

    def forward(self, inputs):
        encoder1 = self.encoder1(inputs)  # (b,128,w,h)
        encoder2 = self.encoder2(self.maxPool(encoder1))  # (b,256,w/2,h/2)
        encoder3 = self.encoder3(self.maxPool(encoder2))  # (b,512,w/4,h/4)
        encoder4 = self.encoder4(self.maxPool(encoder3))  # (b,512,W/8,h/8)
        feature_list = [encoder1, encoder2, encoder3, encoder4]
        return feature_list


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RDPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = nn.Conv2d(in_channels, out_channels, 1)
        self.main = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.dilation = BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)

    def forward(self, inputs):
        res_x = self.res(inputs)
        main_x = self.main(inputs)
        output = self.dilation(res_x + main_x)
        return output


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            BasicConv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(
                middle_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LSSED(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 2, patch_size: int = 4):
        super().__init__()
        self.feature = FeatureExtractor(pretrained)
        self.vit = Vit('B_32_imagenet1k', in_channels=512, pretrained=True, image_size=28, patches=patch_size,
                       num_heads=6)
        self.up = nn.UpsamplingBilinear2d(scale_factor=patch_size)
        self.conv = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1)
        self.dec4 = DecoderBlockV2(1024, 512, 512)
        self.dec3 = DecoderBlockV2(1024, 256, 256)
        self.dec2 = DecoderBlockV2(512, 128, 128)
        self.dec1 = nn.Conv2d(256, 64, kernel_size=1)
        self.res1 = RDPath(128, 128)
        self.res2 = RDPath(256, 256)
        self.res3 = RDPath(512, 512)
        self.res4 = RDPath(512, 512)
        self.seg_head = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        feature_list = self.feature(inputs)
        conv1 = self.res1(feature_list[0])
        conv2 = self.res2(feature_list[1])
        conv3 = self.res3(feature_list[2])
        conv4 = self.res4(feature_list[3])

        center = self.vit(conv4)
        center = self.conv(center)
        dec4 = self.dec4(torch.cat([conv4, self.up(center)], dim=1))
        dec3 = self.dec3(torch.cat([conv3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([conv2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([conv1, dec2], dim=1))
        result = self.seg_head(dec1)
        return [result, dec1]


if __name__ == '__main__':
    test = torch.ones((1, 3, 224, 224)).cuda()
    net = LSSED().cuda()
    a = net(test)
    print(a[0].shape, a[1].shape)
