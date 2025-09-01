import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Sobel = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
Robert = np.array([[0, 0],
                   [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)


def f(x, y):
    return (1 - x) * (1 - y) + 1 / 2 * x * y


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        x = torch.sign(x - y)
        out = (x + 1) / 2
        return out


def Conv(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)



class WeightPredNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
    def forward(self, x):
        return x


class PromptPool(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x):
        return x



def guided_filter(x, y, r=5, eps=1e-8):
    return x


class HAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, x):
        return x


class fusion1(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size, bias):
        super(fusion1, self).__init__()

    def forward(self, x, y=None):
        return x


class fusion2(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, bias):
        super(fusion2, self).__init__()

    def forward(self, x, y):
        return x,y


class HAPFE(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(HAPFE, self).__init__()

    def forward(self, encoder_feature, decoder_feature):
        return encoder_feature, decoder_feature


# EN-DEcoder
class Output(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, output_channel=3, residual=True):
        super(Output, self).__init__()

    def forward(self, x, x_img):
        return x


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, atten):
        super(Encoder, self).__init__()

    def forward(self, x, encoder_outs=None):
            return x


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, residual=True):
        super(Decoder, self).__init__()

    def forward(self, outs):
        return outs


# CPAB
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()

    def forward(self, x):
        return x


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()

    def forward(self, x):
        return x


class CPAB(nn.Module):
    def __init__(self, dim, kernel_size, bias):
        super(CPAB, self).__init__()

    def forward(self, x):
        return x


# MSAB
class MSAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x



class VI_HAPFNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_feat=64, kernel_size=3, reduction=4,
                 bias=False):  # n_feat=80,reduction=4,scale_unetfeats=48,
        super(VI_HAPFNet, self).__init__()

        # Shallow Feature Extraction
        self.inf_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                        CPAB(n_feat, kernel_size, bias),
                                        CPAB(n_feat, kernel_size, bias))
        self.rgb_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                        CPAB(n_feat, kernel_size, bias),
                                        CPAB(n_feat, kernel_size, bias))
        # Multi-Scale Feature Extraction
        self.inf_encoder = Encoder(n_feat, kernel_size, bias, atten=False)
        self.inf_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.rgb_encoder = Encoder(n_feat, kernel_size, bias, atten=True)
        self.rgb_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.conv = Conv(n_feat, output_nc, kernel_size=1, bias=bias)

        # Haze-Aware Prompt-guided Feature Enhancer
        self.inf_structure = HAPFE(n_feat, kernel_size, bias)
        self.rgb_structure = HAPFE(n_feat, kernel_size, bias)

        # Low-Frequency Residual Block
        self.low_freq_compensation = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv(n_feat, n_feat, kernel_size=1),
            nn.PReLU()
        )
        self.low_freq_fuse = Conv(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, rgb, inf):
        # infrared image feature extraction branch
        inf_fea1 = self.inf_layer1(inf)
        inf_encode_feature = self.inf_encoder(inf_fea1)
        inf_decode_feature = self.inf_decoder(inf_encode_feature)
        inf_structure, inf_H = self.inf_structure(inf_encode_feature, inf_decode_feature)

        # visible image feature extraction branch
        rgb_fea1 = self.rgb_layer1(rgb)
        rgb_encode_feature = self.rgb_encoder(rgb_fea1)
        rgb_decode_feature = self.rgb_decoder(rgb_encode_feature)
        rgb_structure, rgb_H = self.rgb_structure(rgb_encode_feature, rgb_decode_feature)

        # Haze-Aware Cross-modal Fusion
        H = (inf_H + rgb_H) / 2
        incons_feature = []
        for i in range(len(rgb_structure)):
            H = F.interpolate(H, size=rgb_structure[i].shape[2:], mode='bilinear', align_corners=False)
            incons_feature.append(f(rgb_structure[i], inf_structure[i]) * (1 + H))
        inf_weight = [None for _ in range(3)]
        for i in range(3):
            inf_weight[i] = incons_feature[i] * inf_structure[i]

        # Infrared-Guided Refined Reconstruction
        rgb_fea2 = self.conv(rgb_decode_feature[0])
        rgb_out1 = rgb_fea2 + rgb
        rgb_fea3 = self.rgb_layer1(rgb_out1)
        rgb_encode_feature_2 = self.rgb_encoder(rgb_fea3, inf_weight)
        rgb_decode_feature_2 = self.rgb_decoder(rgb_encode_feature_2)

        # Low-Frequency Residual Block
        decoded_feat = rgb_decode_feature_2[0]
        low_freq_feat = self.low_freq_compensation(rgb_decode_feature[0])
        fused_feat = torch.cat([decoded_feat, low_freq_feat], dim=1)
        enhanced_feat = self.low_freq_fuse(fused_feat)

        out = self.conv(enhanced_feat)

        return out, rgb_structure, inf_structure, incons_feature, inf_weight


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(DownSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UpSample, self).__init__()
        self.conv = Conv(in_channels, out_channel, 1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class Edge(nn.Module):
    def __init__(self, channel, kernel='sobel'):
        super(Edge, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.kernel_x = Sobel.repeat(channel, 1, 1, 1)
        self.kernel_y = self.kernel_x.permute(0, 1, 3, 2)
        self.kernel_x = nn.Parameter(self.kernel_x, requires_grad=False)
        self.kernel_y = nn.Parameter(self.kernel_y, requires_grad=False)

    def forward(self, current):
        current = F.pad(current, (1, 1, 1, 1), mode='reflect')
        gradient_x = torch.abs(F.conv2d(current, weight=self.kernel_x, groups=self.channel, padding=0))
        gradient_y = torch.abs(F.conv2d(current, weight=self.kernel_y, groups=self.channel, padding=0))
        out = gradient_x + gradient_y
        return out