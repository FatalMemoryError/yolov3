from turtle import forward
from typing import OrderedDict
import torch
from torch import nn

class_nums = 80

class Conv_BN_Leaky(nn.Module):
    def __init__(self, in_num, out_num, k_size, stride, pad):
        super().__init__()
        self.conv2d = nn.Conv2d(in_num, out_num, k_size, stride, pad)
        self.BatchNorm2d = nn.BatchNorm2d(out_num)
        self.Leaky_ReLU = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.BatchNorm2d(x)
        x = self.Leaky_ReLU(x)
        return x

class Res_Unit(nn.Module):
    def __init__(self, in_num, out_num):
        super().__init__()
        self.CBL1 = Conv_BN_Leaky(in_num, out_num, k_size=1, stride=1, pad=0)
        self.CBL2 = Conv_BN_Leaky(out_num, in_num, k_size=3, stride=1, pad=1)
    
    def forward(self, x):
        res = x
        x = self.CBL1(x)
        x = self.CBL2(x)
        x = x + res
        return x

class ResX(nn.Module):
    def __init__(self, in_num, out_num, block_num):
        super().__init__()
        self.block_num = block_num
        self.CBL = Conv_BN_Leaky(in_num, out_num, k_size=3, stride=2, pad=1)
        self.Res_X = self.make_layers(out_num, in_num, block_num)

    def forward(self, x):
        x = self.CBL(x)
        x = self.Res_X(x)
        return x
    
    def make_layers(self, in_num, out_num, block_num):
        layers = []
        for i in range(block_num):
            layers.append(Res_Unit(in_num, out_num))
        return nn.Sequential(*layers)

class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.CBL = Conv_BN_Leaky(3, 32, k_size=3, stride=1, pad=1)
        self.Res1_1 = ResX(32, 64, 1)
        self.Res2_2 = ResX(64, 128, 2)
        self.Res3_8 = ResX(128, 256, 8)
        self.Res4_8 = ResX(256, 512, 8)
        self.Res5_4 = ResX(512, 1024, 4)
    
    def forward(self, x):
        x = self.CBL(x)
        x = self.Res1_1(x)
        x = self.Res2_2(x)
        x = self.Res3_8(x)
        out3 = x
        x = self.Res4_8(x)
        out4 = x
        x = self.Res5_4(x)
        out5 = x
        return out3, out4, out5

class CBL_5(nn.Module):
    def __init__(self, in_num):
        super().__init__()
        self.CBL1 = Conv_BN_Leaky(in_num, in_num//2, k_size=1, stride=1, pad=0)
        self.CBL2 = Conv_BN_Leaky(in_num//2, in_num, k_size=3, stride=1, pad=1)
        self.CBL3 = Conv_BN_Leaky(in_num, in_num//2, k_size=1, stride=1, pad=0)
        self.CBL4 = Conv_BN_Leaky(in_num//2, in_num, k_size=3, stride=1, pad=1)
        self.CBL5 = Conv_BN_Leaky(in_num, in_num//2, k_size=1, stride=1, pad=0)

    def forward(self, x):
        x = self.CBL1(x)
        x = self.CBL2(x)
        x = self.CBL3(x)
        x = self.CBL4(x)
        x = self.CBL5(x)
        return x

class Yolo_head(nn.Module):
    def __init__(self, in_num):
        super().__init__()
        self.CBL = Conv_BN_Leaky(in_num, in_num*2, k_size=3, stride=1, pad=1)
        self.conv1 = nn.Conv2d(in_num*2, 255, kernel_size=1, stride=1, pad=0)
    
    def forward(self, x):
        x = self.CBL(x)
        x = self.conv1(x)
        return x


class Yolov3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DarkNet53()
        self.last_layer1 = nn.Sequential(OrderedDict([
                                         ('CBL5_1', CBL_5(1024)),
                                         ('Yolo_head_1', Yolo_head(1024//2))
                                        ]))
        self.last_layer2_cbl = Conv_BN_Leaky(512, 256, k_size=1, stride=1, pad=0)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = nn.Sequential(OrderedDict([
                                         ('CBL5_2', CBL_5(768)),
                                         ('Yolo_head_2', Yolo_head(768//2))
                                        ]))
        self.last_layer3_cbl = Conv_BN_Leaky(256, 128, k_size=1, stride=1, pad=0)
        self.last_layer3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer3 = nn.Sequential(OrderedDict([
                                         ('CBL5_3', CBL_5(384)),
                                         ('Yolo_head_2', Yolo_head(384//2))
                                        ]))
    
    def forward(self, x):
        x3, x4, x5 = self.backbone(x)
        layer1_branch = self.last_layer1[0](x5)
        layer1_out = self.last_layer1[1](layer1_branch)   

        layer2_in = self.last_layer2_cbl(layer1_branch)
        layer2_in = self.last_layer2_upsample(layer2_in)
        layer2_in = torch.concat([layer2_in, x4], 1)
        layer2_branch = self.last_layer2[0](layer2_in)
        layer2_out = self.last_layer2(layer2_branch)
        
        layer3_in = self.last_layer3_cbl(layer2_branch)
        layer3_in = self.last_layer3_upsample(layer3_in)
        layer3_in = torch.concat([layer3_in, x3], 1)
        layer3_out = self.last_layer3(layer3_in)
        return layer1_out, layer2_out, layer3_out