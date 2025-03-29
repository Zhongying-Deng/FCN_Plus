import math
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module('GlobalShift')
class GlobalShiftV2Portion(nn.Module):

    def __init__(self, scale=1, portion=0.5):
        super(GlobalShiftV2Portion, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   self.scale, h // self.scale, self.scale,
                   w // self.scale)  # dim=7
        x = x.permute(0, 1, 2, 4, 6, 3,
                      5).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  # dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).cuda()  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 4, 2, 5,
                                       3).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            if math.floor(shift /
                          (self.scale * self.scale)) != shift / (self.scale *
                                                                 self.scale):
                shift = math.floor(shift / (self.scale * self.scale)) * (
                    self.scale * self.scale)
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1)

        return x


@PLUGIN_LAYERS.register_module('LocalShift')
class LocalShiftV2Portion(nn.Module):

    def __init__(self, scale=1, portion=0.5):
        super(LocalShiftV2Portion, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   h // self.scale, self.scale, w // self.scale,
                   self.scale)  # dim=7
        x = x.permute(0, 1, 2, 3, 5, 4,
                      6).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  # dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).cuda()  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 2, 4, 3,
                                       5).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1)

        return x


@PLUGIN_LAYERS.register_module('GlobalShiftResidual')
class GlobalShiftV2PortionResidual(nn.Module):

    def __init__(self, scale=1, portion=0.5):
        super(GlobalShiftV2PortionResidual, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   self.scale, h // self.scale, self.scale,
                   w // self.scale)  # dim=7
        x = x.permute(0, 1, 2, 4, 6, 3,
                      5).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  # dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).cuda()  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 4, 2, 5,
                                       3).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            if math.floor(shift /
                          (self.scale * self.scale)) != shift / (self.scale *
                                                                 self.scale):
                shift = math.floor(shift / (self.scale * self.scale)) * (
                    self.scale * self.scale)
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1) + x

        return x


@PLUGIN_LAYERS.register_module('LocalShiftResidual')
class LocalShiftV2PortionResidual(nn.Module):

    def __init__(self, scale=1, portion=0.5):
        super(LocalShiftV2PortionResidual, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   h // self.scale, self.scale, w // self.scale,
                   self.scale)  # dim=7
        x = x.permute(0, 1, 2, 3, 5, 4,
                      6).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  # dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).cuda()  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 2, 4, 3,
                                       5).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1) + 1

        return x


@PLUGIN_LAYERS.register_module('ASPPShiftV2')
class ASPPShiftV2(nn.Module):

    def __init__(self, scale=1):
        super(ASPPShiftV2, self).__init__()
        self.scale = scale
        self.index = (
            torch.arange(0, scale * scale).view(1, scale * scale) +
            torch.arange(0, scale * scale).view(scale * scale, 1)).fmod(
                scale * scale).long()

    def forward(self, x):
        b, c, h, w = x.shape
        # print(x.shape)
        # pooled_x = F.adaptive_avg_pool2d(x, self.scale) # [b, c, self.scale, self.scale]
        # if self.scale != 1:
        # pooled_x = pooled_x.view(b, self.scale*self.scale, c//(self.scale*self.scale), self.scale*self.scale)
        # index = self.index.view(1, self.scale * self.scale, 1, self.scale * self.scale).repeat(b, 1, c // (self.scale * self.scale), 1).cuda()
        # pooled_x = torch.gather(pooled_x, dim=3, index=index)
        # pooled_x = pooled_x.view(b, c, self.scale, self.scale)
        x = x.view(b, self.scale * self.scale, c // (self.scale * self.scale),
                   self.scale, h // self.scale, self.scale,
                   w // self.scale)  # dim=7
        x = x.permute(0, 1, 2, 4, 6, 3,
                      5).contiguous().view(b, self.scale * self.scale,
                                           c // (self.scale * self.scale),
                                           h // self.scale, w // self.scale,
                                           self.scale * self.scale)  #dim=6
        index = self.index.view(1, self.scale * self.scale, 1, 1, 1,
                                self.scale * self.scale).repeat(
                                    b, 1, c // (self.scale * self.scale),
                                    h // self.scale, w // self.scale,
                                    1).cuda()  #dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale, w // self.scale, self.scale,
                   self.scale).permute(0, 1, 4, 2, 5,
                                       3).contiguous().view(b, c, h, w)
        # pooled_x = pooled_x.view(b, self.scale*self.scale, c//(self.scale*self.scale), self.scale*self.scale).permute(0,3,2,1).contiguous().view(b, c, self.scale, self.scale)
        # pooled_x = self.pooled_redu_conv(pooled_x)
        # pooled_x = F.upsample(pooled_x, size=(h, w), **up_kwargs)

        return x


@PLUGIN_LAYERS.register_module('GlobalShift2d')
class GlobalShift2dV2Portion(nn.Module):

    def __init__(self, scale=(4, 4), portion=0.5):
        super(GlobalShift2dV2Portion, self).__init__()
        self.scale = scale
        self.portion = portion
        self.index = (
            torch.arange(0, scale[0] * scale[1]).view(1, scale[0] * scale[1]) +
            torch.arange(0, scale[0] * scale[1]).view(
                scale[0] * scale[1], 1)).fmod(scale[0] * scale[1]).long()

    def shift_scale(self, x):
        b, c, h, w = x.shape

        x = x.view(b, self.scale[0] * self.scale[1],
                   c // (self.scale[0] * self.scale[1]), self.scale[0],
                   h // self.scale[0], self.scale[1],
                   w // self.scale[1])  # dim=7
        x = x.permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(
            b, self.scale[0] * self.scale[1],
            c // (self.scale[0] * self.scale[1]), h // self.scale[0],
            w // self.scale[1], self.scale[0] * self.scale[1])  # dim=6
        index = self.index.view(1, self.scale[0] * self.scale[1], 1, 1, 1,
                                self.scale[0] * self.scale[1]).repeat(
                                    b, 1, c // (self.scale[0] * self.scale[1]),
                                    h // self.scale[0], w // self.scale[1],
                                    1).cuda()  # dim=6
        x = torch.gather(x, dim=5, index=index)
        x = x.view(b, c, h // self.scale[0], w // self.scale[1], self.scale[0],
                   self.scale[1]).permute(0, 1, 4, 2, 5,
                                          3).contiguous().view(b, c, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        if self.portion > 0:
            shift = int(self.portion * c)
            if math.floor(shift / (self.scale[0] * self.scale[1])) != shift / (
                    self.scale[0] * self.scale[1]):
                shift = math.floor(shift / (self.scale[0] * self.scale[1])) * (
                    self.scale[0] * self.scale[1])
            keep_x, shift_x = torch.split(
                x, split_size_or_sections=[c - shift, shift], dim=1)
            shift_x = self.shift_scale(shift_x)
            x = torch.cat([keep_x, shift_x], dim=1)

        return x