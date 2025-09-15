import torch
import torch.nn as nn
import torch.nn.functional as F

from model_MCANet.MCAM import MCAM
from model_MCANet.ASPP import ASPP
from model_MCANet.ResNet import resnet101
from models.common import *

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MACANet(nn.Module):
    def __init__(self, pretrained = True, backbone = 'ResNet101', att_type=None, num_feats = 32, kernel_size = 3, scale = 4):
        super(MACANet, self).__init__()
        # print(num_classes)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')
        self.encoder = EncoderBlock(pretrained, backbone, att_type=att_type)
        self.decoder = Super_resolution(num_feats, kernel_size, scale)


    def forward(self, sar_img, opt_img, depth_img):
        sar_img_up = self.bicubic(sar_img)
        opt_sar_low_high_features = self.encoder.forward(sar_img_up, opt_img, depth_img)
        classification = self.decoder(opt_sar_low_high_features)

        return classification + sar_img_up
class EncoderBlock(nn.Module):
    def __init__(self, pretrained = True, backbone = 'ResNet101', num_classes=1000, att_type=None):
        super(EncoderBlock, self).__init__()
        if backbone == 'ResNet101':
            self.SAR_resnet = resnet101(pretrained, type='sar', num_classes=num_classes, att_type=att_type)
            self.OPT_resnet = resnet101(pretrained, type='opt', num_classes=num_classes, att_type=att_type)
            self.Depth_resnet = resnet101(pretrained, type='sar', num_classes=num_classes, att_type=att_type)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use ResNet101.'.format(backbone))

        self.MCAM_low = MCAM(in_channels=256)
        self.MCAM_high = MCAM(in_channels=2048)
        self.ASPP = ASPP(in_channels=2304, atrous_rates=[6, 12, 18])
        self.conv1 = conv1x1(2048, 256)
        self.conv2 = conv1x1(512, 48)

    def forward(self, sar_img, opt_img, depth_img):
        sar_feats = self.SAR_resnet.forward(sar_img)
        opt_feats = self.OPT_resnet.forward(opt_img)
        depth_feats = self.Depth_resnet.forward(depth_img)

        sar_low_feat = sar_feats[1]
        sar_high_feat = sar_feats[4]
        sar_final_feat = self.conv1(sar_feats[4])
        opt_low_feat = opt_feats[1]
        opt_high_feat = opt_feats[4]
        opt_final_feat = self.conv1(opt_feats[4])
        depth_low_feat = depth_feats[1]
        depth_high_feat = depth_feats[4]
        depth_final_feat = self.conv1(depth_feats[4])

        low_level_features = self.MCAM_low(sar_low_feat, opt_low_feat)#8 256 50 50
        high_level_features = self.MCAM_high(sar_high_feat, opt_high_feat)#8 2048 7 7

        # low_level_sar_opt = torch.cat([sar_low_feat, opt_low_feat], 1)
        # high_level_sar_opt = torch.cat([sar_final_feat, opt_final_feat], 1)
        low_level_sar_opt = sar_low_feat + opt_low_feat + depth_low_feat
        low_sar_opt_features = torch.cat([low_level_sar_opt, low_level_features], 1)#256+256
        high_level_sar_opt = sar_final_feat + opt_final_feat + depth_final_feat
        high_sar_opt_features =  torch.cat([high_level_sar_opt, high_level_features], 1)#2048+256

        # low_sar_opt_features = torch.cat([low_level_sar_opt, low_level_features], 1)
        # high_sar_opt_features = torch.cat([high_level_sar_opt, high_level_features], 1)
        # low_sar_opt_features = low_level_sar_opt + low_level_features
        # high_sar_opt_features = high_level_sar_opt + high_level_features

        low_sar_opt_features = self.conv2(low_sar_opt_features)#8 512 50 50
        high_sar_opt_features = self.ASPP(high_sar_opt_features)#8 2304 7 7
        high_sar_opt_features = F.interpolate(high_sar_opt_features, size=(64, 64), mode='bilinear', align_corners=False)

        opt_sar_low_high_features = torch.cat([high_sar_opt_features, low_sar_opt_features], 1)

        return opt_sar_low_high_features
class DecoderBlock(nn.Module):
    def __init__(self, num_class):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, num_class, kernel_size=1)
        )
    def forward(self, opt_sar_low_high_features):
        final_class = self.conv(opt_sar_low_high_features)
        final_img = F.interpolate(final_class, size=(256, 256), mode='bilinear', align_corners=False)

        return final_img

class Super_resolution(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(Super_resolution, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        my_tail = [
            ResidualGroup(
                default_conv, 128, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 128, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(128, 128, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(128, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)


        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, dp):
        dp1 = self.conv_a(dp)#dp 8 304 50 50
        tail_in = self.upsampler(dp1)#dp 8 128 50 50
        out = self.last_conv(self.tail(tail_in))
        return out

if __name__ == '__main__':
    upscale = 4
    # height = (1024 // upscale // window_size + 1) * window_size
    # width = (720 // upscale // window_size + 1) * window_size
    height = 255
    width = 255
    model = MACANet(num_feats=32, kernel_size=3, scale=upscale)
    print(model)

    lr = torch.randn(8, 1, 64, 64)
    depth = torch.randn(8, 1, 256, 256)
    rgb = torch.randn(8, 3, 256, 256)

    x = model(lr, rgb, depth)#net((guidance, lr))
    print(x.shape)
