from torch import nn
from torch.nn import functional as F
import torch
from src.resnet import resnet18
import functools

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class ConvReluBn(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ReluBn(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        return x

class DecoderBlockBN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockBN, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            ConvReluBn(in_channels, middle_channels),
            ConvReluBn(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ResnetUNET(nn.Module):

    def __init__(self, num_classes=2, num_filters=32, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = resnet18()
        self.relu = nn.ReLU(inplace=True)

        self.layer0_conv1 = self.encoder.conv1
        self.layer0_bn1 = self.encoder.bn1
        self.layer0_relu1 = self.encoder.relu
        self.layer0_maxpool1 = self.encoder.maxpool

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockBN(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockBN(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockBN(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockBN(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockBN(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockBN(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvReluBn(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(0.5)


    def forward(self, x):
        x = self.layer0_conv1(x)
        x = self.layer0_bn1(x)
        x = self.layer0_relu1(x)
        x = self.layer0_maxpool1(x)

        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4)
        center = self.center(self.pool(conv5))
        center = self.drop(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        pred_mask = self.final(dec0)

        return pred_mask
