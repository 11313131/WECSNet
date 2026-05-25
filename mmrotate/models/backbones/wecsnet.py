import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import pytorch_wavelets as ptw
import os
import copy
from mmcv.cnn import build_norm_layer
from math import log
import numpy
import matplotlib.pyplot as plt
from ..builder import ROTATED_BACKBONES
try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]
        x = self.conv(x)  # x = [B, 2C, H, W]
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]
        return x

def show_feature(out, save_name='feature_map'):
    out_cpu = out.cpu()
    feature_map = out_cpu.detach().numpy()
    im = np.squeeze(feature_map)
    im = np.transpose(im, [1, 2, 0])
    # 处理可能通道数不足的情况
    C = min(im.shape[-1], 24)
    for c in range(C):
        plt.subplot(4, 6, c+1)
        plt.axis('off')
        plt.imshow(im[:, :, c], cmap='Blues')
    plt.savefig(f'{save_name}.png', dpi=300)
    plt.clf()
    plt.close()

def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


# Multi-scale Cross-Strip Convolution (MCSConv)
class MCSConv(nn.Module):
    def __init__(self, in_dim, norm_layer, act_layer, out_dim=None, h_kernel_size=[3, 7, 11],
                 v_kernel_size=[3, 7, 11], expansion: float = 1.0):
        super(MCSConv , self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = make_divisible(int(out_dim * expansion), 8)
        self.conv5 = nn.Conv2d(in_dim, hidden_dim, kernel_size=5, padding=2, dilation=1, padding_mode='reflect', groups=hidden_dim)
        self.h_conv1 = nn.Conv2d(hidden_dim, hidden_dim, (1, h_kernel_size[0]), 1, (0, h_kernel_size[0] // 2),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.v_conv1 = nn.Conv2d(hidden_dim, hidden_dim, (v_kernel_size[0], 1), 1, (v_kernel_size[0] // 2, 0),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.h_conv2 = nn.Conv2d(hidden_dim, hidden_dim, (1, h_kernel_size[1]), 1, (0, h_kernel_size[1] // 2),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.v_conv2 = nn.Conv2d(hidden_dim, hidden_dim, (v_kernel_size[1], 1), 1, (v_kernel_size[1] // 2, 0),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.h_conv3 = nn.Conv2d(hidden_dim, hidden_dim, (1, h_kernel_size[2]), 1, (0, h_kernel_size[2] // 2),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.v_conv3 = nn.Conv2d(hidden_dim, hidden_dim, (v_kernel_size[2], 1), 1, (v_kernel_size[2] // 2, 0),
                                 padding_mode='reflect', groups=hidden_dim, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, dilation=1)
        self.SE2 = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, dilation=1)
        self.SE3 = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, dilation=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, out_dim)[1],
            act_layer())

    def forward(self, x):
        x1 = self.conv5(x)
        x1_h = self.h_conv1(x1)
        x1_v = self.v_conv1(x1)
        x1_hv = x1_h + x1_v
        x2 = self.conv5(x)
        x2_h = self.h_conv2(x2)
        x2_v = self.v_conv2(x2)
        x2_hv = x2_h + x2_v
        x3 = self.conv5(x)
        x3_h = self.h_conv3(x3)
        x3_v = self.v_conv3(x3)
        x3_hv = x3_h + x3_v
        y1_weight = self.SE1(self.gap(x1_hv))
        y2_weight = self.SE2(self.gap(x2_hv))
        y3_weight = self.SE3(self.gap(x3_hv))
        # 拼接全局信息
        weight = torch.cat([y1_weight, y2_weight, y3_weight], 2)
        # 计算权重
        weight = self.softmax(self.Sigmoid(weight))
        # 调整权重维度
        y1_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 2], 2)
        # 加权求和
        x_att = y1_weight * x1_hv + y2_weight * x2_hv + y3_weight * x3_hv
        out = self.conv_block(x_att) + x
        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer):
        super().__init__()
        gaussian = self.gaussian_kernel(size, sigma)  # 生成高斯核
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        with torch.no_grad():
            self.gaussian.weight.copy_(gaussian.repeat(dim, 1, 1, 1))
        self.gaussian.weight.requires_grad = False
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def forward(self, x):
        # 1. 应用高斯滤波
        gaussian_out = self.gaussian(x)  # 高斯平滑后的特征
        # 2. 归一化和激活,训练更稳定
        out = self.act(self.norm(gaussian_out))
        return out

    #  高斯核生成函数
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # 二维高斯函数
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()  # 归一化：确保卷积后图像亮度不变


class Conv(nn.Module):
    def __init__(self, c1, c2, norm_layer, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = build_norm_layer(norm_layer, c2)[1]
        self.act = nn.ReLU()if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GSConv(nn.Module):
    def __init__(self, c1, c2, norm_layer, k=1, s=1, g=1, act=True):
        super().__init__()
        # c1: 输入通道数,c2: 输出通道数,g: 分组卷积的组数
        c_ = c2 // 2  # 将输出通道数分成两半
        self.cv1 = Conv(c1, c_, norm_layer, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, norm_layer, 3, 1, None, 4, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])

    def forward(self, x):
        out = self.block(x)
        return out


class HLFE(nn.Module):
    def __init__(self, in_channels, stage, norm_layer, act_layer, wavelet='haar'):
        super(HLFE, self).__init__()
        t = int(abs((log(in_channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.stage = stage
        self.wavelet = wavelet
        self.dwt = ptw.DWTForward(J=1, wave=wavelet, mode='zero')
        self.idwt = ptw.DWTInverse(wave=wavelet, mode='zero')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if stage == 0:
            # 3×3的GSConv处理 HL, LH, HH 三个高频分量, 用于浅层特征
            self.conv_hl = GSConv(in_channels, in_channels, norm_layer, g=4)
            self.conv_lh = GSConv(in_channels, in_channels, norm_layer, g=4)
            self.conv_hh = GSConv(in_channels, in_channels, norm_layer, g=4)
        else:
            # 高斯核卷积处理 LL 低频分量, 用于深层特征, 用较大的卷积核
            self.gaussian_ll = Gaussian(in_channels, 9, 2.0, norm_layer, act_layer)
        self.conv_extra = Conv_Extra(in_channels, norm_layer, act_layer)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, in_channels)[1],
            act_layer())
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, in_channels)[1]

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: 增强后的高频特征 [B, C, H, W]
        """
        # 1. DWT 分解, 返回低频和高频分量
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x_float = x.float()
            yl, yh = self.dwt(x_float)
            y_hl = yh[0][:, :, 0, ::]  # float32
            y_lh = yh[0][:, :, 1, ::]
            y_hh = yh[0][:, :, 2, ::]
            y_ll = yl
            # 卷积层在 float32 下计算
            if self.stage == 0:
                # GSConv处理每个高频分量
                hl_enhanced = self.conv_hl(y_hl)
                lh_enhanced = self.conv_lh(y_lh)
                hh_enhanced = self.conv_hh(y_hh)
                # 重构三个增强后的高频分量，形成形状为 [B, C, 3, H/2, W/2] 的张量
                enhanced_yh = torch.stack([hl_enhanced, lh_enhanced, hh_enhanced], dim=2)
                # IDWT 重构:使用原始低频分量 yl 和增强后的高频分量进行重构
                reconstructed = self.idwt((y_ll, [enhanced_yh]))
                reconstructed = reconstructed.to(orig_dtype)  # 转回原始类型
            else:
                # 高斯核卷积处理低频分量
                ll_enhanced = self.gaussian_ll(y_ll)
                # 将原始高频分量堆叠成1个张 量，形状为 [B, C, 3, H/2, W/2]
                original_yh = torch.stack([y_hl, y_lh, y_hh], dim=2)
                # IDWT 重构:使用原始高频分量和增强后的低频分量 yl 进行重构
                reconstructed = self.idwt((ll_enhanced, [original_yh]))
                reconstructed = reconstructed.to(orig_dtype)  # 转回原始类型
        # 尺寸对齐
        if reconstructed.shape != x.shape:
            _, _, h, w = x.shape
            reconstructed = reconstructed[:, :, :h, :w]
        # 后续操作在混合精度下进行
        reconstructed = reconstructed + x
        reconstructed = self.conv_extra(reconstructed)
        x_out = reconstructed * x  + x
        y_out = self.conv_block(x_out)
        out = self.avg_pool(y_out)
        out = self.conv1d(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out) * y_out
        return self.norm(out + x)


class WECS_Module(nn.Module):
    def __init__(self, dim, stage, norm_layer, act_layer):
        super().__init__()
        self.norm1 = build_norm_layer(norm_layer, dim)[1]
        self.norm2 = build_norm_layer(norm_layer, dim)[1]
        self.MCSConv  = MCSConv (dim, norm_layer, act_layer)
        self.HLFE = HLFE(dim, stage, norm_layer, act_layer)

    def forward(self, x):
        identity1 = x.clone()
        x = self.norm1(x)
        x = self.MCSConv(x)
        x = x + identity1
        identity2 = x.clone()
        x = self.norm2(x)
        x = self.HLFE(x)
        return x + identity2


class WECS_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]
        self.mlp = nn.Sequential(*mlp_layer)
        self.wecs_module = WECS_Module(dim, stage, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor) -> Tensor:
        #show_feature(x)
        x_att = self.wecs_module(x)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super().__init__()

        blocks_list = [
            WECS_Block(
                dim=dim,
                stage=stage,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class Stem(nn.Module):
    """Stem layer"""
    def __init__(
            self,
            in_channels,
            out_channels,
            act_layer,
            norm_layer,
            expansion: float = 1.0
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(norm_layer, hidden_channels)[1]
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(norm_layer, out_channels)[1]
        self.act2 = act_layer()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

# Wavelet-Enhanced Cross-Strip Network
@ROTATED_BACKBONES.register_module()
class WECSNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes  # 分类任务才需要类别数
        self.num_stages = len(depths)  # 阶段数 = depths的长度
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))  # 最终特征维度

        if stem_dim == 96:
            act_layer = nn.ReLU
        # Stem模块初始化
        self.Stem = Stem(in_channels=in_chans, out_channels=stem_dim, act_layer=act_layer, norm_layer=norm_layer)

        # stochastic depth decay rule：随机深度配置
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):  # 遍历每个阶段
            # 1. 添加BasicStage
            stage = BasicStage(dim=int(stem_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            stages_list.append(stage)
            # 2. 阶段间下采样（除最后一阶段外）
            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat  # 模式选择
        if self.fork_feat:
            self.forward = self.forward_det  # 设置为特征提取模式
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls  # 设置为分类头模式
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.Stem(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.Stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # return outs
        return tuple(outs)
