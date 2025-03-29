import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead


class SqueezeBodyEdge(nn.Module):

    def __init__(self, in_channels, conv_cfg, norm_cfg, act_cfg,
                 align_corners):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.align_corners = align_corners
        self.down = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                3,
                groups=in_channels,
                stride=2,
                bias=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels,
                in_channels,
                3,
                groups=in_channels,
                stride=2,
                bias=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )

        self.flow_make = nn.Conv2d(
            in_channels * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = resize(
            seg_down, size=size)
            # seg_down, size=size, mode="bilinear", align_corners=True)
        # align_corners=self.align_corners)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, feat, flow, size):
        out_h, out_w = size
        n, c, h, w = feat.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(feat).to(feat.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(feat).to(feat.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(feat, grid)
        return output


@HEADS.register_module()
class ASPPDecoupleSegHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self,
                 c1_in_channels,
                 c1_channels,
                 dilations=(1, 6, 12, 18),
                 edge_attention_thresh=0.8,
                 loss_body_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.2),
                 loss_edge_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.2),
                 **kwargs):
        super(ASPPDecoupleSegHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.edge_attention_thresh = edge_attention_thresh
        self.loss_body_decode = build_loss(loss_body_decode)
        self.loss_edge_decode = build_loss(loss_edge_decode)
        # self.loss_body_decode = None
        # self.loss_edge_decode = None
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                bias=False,  # different
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.bottleneck = ConvModule(
        #     (len(dilations) + 1) * self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        # self.c1_bottleneck = ConvModule(
        #     c1_in_channels,
        #     c1_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        self.c1_bottleneck = ConvModule(
            c1_in_channels,
            c1_channels,
            1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        # body_edge module
        self.squeeze_body_edge = SqueezeBodyEdge(
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        # fusion different edge part
        # self.edge_fusion = ConvModule(
        #     self.channels + c1_channels,
        #     self.channels,
        #     1,
        #     bias=False,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        self.edge_fusion = ConvModule(
            self.channels + c1_channels,
            self.channels,
            1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        # DSN for edge out
        # self.edge_seg = ConvModule(
        #         self.channels,
        #         c1_channels,
        #         3,
        #         padding=1,
        #         bias=False,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        # self.conv_edge_seg = nn.Conv2d(c1_channels, 1, kernel_size=1, bias=False)

        # DSN for seg body out
        # self.body_seg = ConvModule(
        #         self.channels,
        #         self.channels,
        #         3,
        #         padding=1,
        #         bias=False,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        # self.conv_body_seg = nn.Conv2d(
        #         self.channels, self.num_classes, kernel_size=1, bias=False)

        # Final segmentation out
        self.final_seg = nn.Sequential(ConvModule(
                self.channels * 2,
                self.channels,
                3,
                padding=1,
                bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.conv_seg = nn.Conv2d(
                self.channels, self.num_classes, kernel_size=1, bias=False)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        # aspp_outs = [
        #     resize(
        #         self.image_pool(x),
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:])
        ]

        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_fused = self.bottleneck(aspp_outs)
        seg_body, seg_edge = self.squeeze_body_edge(aspp_fused)
        c1_output = self.c1_bottleneck(inputs[0])

        # seg_edge = self.edge_fusion(
        #     torch.cat([
        #         resize(
        #             seg_edge,
        #             size=c1_output.shape[2:],
        #             mode='bilinear',
        #             align_corners=self.align_corners), c1_output
        #     ],
        #               dim=1))
        seg_edge = self.edge_fusion(
            torch.cat([
                resize(
                    seg_edge,
                    size=c1_output.shape[2:]), c1_output
            ],
                      dim=1))

        # edge_seg_out = self.dsn_edge_seg(seg_edge)
        # body_seg_out = self.dsn_body_seg(seg_body)

        # seg_out = seg_edge + resize(
        #     seg_body,
        #     size=c1_output.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        seg_out = seg_edge + resize(
            seg_body,
            size=c1_output.shape[2:])

        # aspp_fused = resize(
        #     aspp_fused,
        #     size=c1_output.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        aspp_fused = resize(
            aspp_fused,
            size=c1_output.shape[2:])

        seg_out = torch.cat([aspp_fused, seg_out], dim=1)
        final_seg_out = self.dsn_final_seg(seg_out)

        # return final_seg_out, body_seg_out, edge_seg_out
        return final_seg_out, None, None

    def dsn_final_seg(self, feat):
        feat = self.final_seg(feat)
        """Final feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def dsn_body_seg(self, feat):
        feat = self.body_seg(feat)
        """Body feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_body_seg(feat)
        return output

    def dsn_edge_seg(self, feat):
        feat = self.edge_seg(feat)
        """Edge feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_edge_seg(feat)
        return output

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``final_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        """Compute ``final_seg``, ``body_seg``, ``edge_seg`` loss."""
        final_seg_out, body_seg_out, edge_seg_out = seg_logit
        # gt_semantic_seg, soft_one_hot, edge_map = seg_label
        gt_semantic_seg = seg_label
        loss = dict()
        loss.update(
            super(ASPPDecoupleSegHead, self).losses(final_seg_out,
                                                    gt_semantic_seg))
        # body_loss = self.loss_body_decode(body_seg_out, soft_one_hot)
        # loss['loss_body'] = body_loss
        # edge_loss = self.loss_edge_decode(edge_seg_out, edge_map)
        # loss['loss_edge'] = edge_loss
        # loss.update(
        #     add_prefix(
        #         self.edge_attention(final_seg_out, gt_semantic_seg,
        #                             edge_seg_out)), 'edge_attention')

        return loss

    def edge_attention(self, final_seg_out, gt_semantic_seg, edge_seg_out):
        filler = torch.full_like(gt_semantic_seg, self.ignore_index)
        target = torch.where(
            edge_seg_out.max(1)[0] > self.edge_attention_thresh,
            gt_semantic_seg, filler)
        return super(ASPPDecoupleSegHead, self).losses(final_seg_out, target)
