#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import eulid_to_homo


__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, inference_mode = False):
        super().__init__()
        self.convs = nn.Sequential(
            MobileOneBlock(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=1,
                            inference_mode=inference_mode,
                            use_se=False,
                            num_conv_branches=1),
            MobileOneBlock(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            inference_mode=inference_mode,
                            use_se=False,
                            num_conv_branches=1),
        )
    def __call__(self, x):
        return self.convs(x)
    

class UpHead(nn.Module):
    def __init__(self,width_multipliers,inference_mode):
        super().__init__()
        self.proj4 = nn.Sequential(
            MobileConv(int(512 * width_multipliers[3]), int(256 * width_multipliers[2]), stride=1, inference_mode=inference_mode),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.proj3 = nn.Sequential(
            MobileConv(int(256 * width_multipliers[2]), int(128 * width_multipliers[1]), stride=1, inference_mode=inference_mode),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.proj2 = nn.Sequential(
            MobileConv(int(128 * width_multipliers[1]), 256, stride=1, inference_mode=inference_mode),
            # nn.UpsamplingBilinear2d(scale_factor=2)
            MobileConv(256, 128, stride=1, inference_mode=inference_mode),
        )
        # self.proj1 = nn.Sequential(
        #     MobileConv(int(64 * width_multipliers[0]), min(64,int(64 * width_multipliers[0])), stride=1, inference_mode=inference_mode),
        #     # nn.UpsamplingBilinear2d(scale_factor=2)
        # )
        # self.proj0 = nn.Sequential(
        #     MobileConv( min(64,int(64 * width_multipliers[0])), 64, stride=1, inference_mode=inference_mode),
        # )

    def forward(self, x2,x3,x4):
        elt3 = x3 + self.proj4(x4) 
        elt2 = x2 + self.proj3(elt3)
        out = self.proj2(elt2)
        # elt1 = x1 + self.proj2(elt2)
        # elt0 = x0 + self.proj1(elt1)
        # out = self.proj0(elt0)
        return out


class Pose2d(nn.Module):
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ 
        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        # self.neck = nn.Sequential(
        #     MobileConv(int(512 * width_multipliers[3]), 512, stride=1, inference_mode=inference_mode),
        #     MobileConv(512, 256, stride=1, inference_mode=inference_mode),
        #     # nn.UpsamplingBilinear2d(scale_factor=2),
        #     MobileConv(256, 256, stride=2, inference_mode=inference_mode),
        #     MobileConv(256, 128, stride=1, inference_mode=inference_mode),
        #     MobileConv(128, 128, stride=2, inference_mode=inference_mode),
        #     MobileConv(128, 64, stride=1, inference_mode=inference_mode),
        # )
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.linear = nn.Linear(64, num_classes*2)

        self.upstage = UpHead(width_multipliers, inference_mode)
        self.hm_head = nn.Sequential(
            MobileConv( 128, num_classes, stride=1, inference_mode=inference_mode),
            nn.Softmax2d(),
        )
        self.reg_head = nn.Sequential(
            MobileConv( 128, num_classes*2, stride=1, inference_mode=inference_mode),
        )
        self.freeze()

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def freeze(self):
        for layer in [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in layer.parameters():
                param.requires_grad = False
        
        for layer in [self.upstage, self.hm_head,self.reg_head]:
            for param in layer.parameters():  
                param.requires_grad = True
            layer.train()
                

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        feature = self.upstage(x2,x3,x4)
        # feature = self.neck(x)
        # out = self.gap(feature)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return self.hm_head(feature), self.reg_head(feature), feature



def pose2d_model(num_classes: int = 1000, inference_mode: bool = False) -> nn.Module:
    return Pose2d(num_classes=num_classes, inference_mode=inference_mode,
                     width_multipliers = (3.0, 3.5, 3.5, 4.0),use_se=True)


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model



class ProbTri(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = pose2d_model(num_classes=cfg['num_keypoints'])
        self.embed = nn.Linear(4,128)
        self.pose0 = nn.Sequential(
            nn.Linear(cfg['num_keypoints']*256, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.pose1 = nn.Sequential(
            nn.Linear(cfg['num_views']*256, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,cfg['num_keypoints']*3),
        )
        self.freeze()

    def freeze(self):
        for layer in [self.backbone]:
            for param in layer.parameters():
                param.requires_grad = False
        
        for layer in [self.embed, self.pose0, self.pose1]:
            for param in layer.parameters():  
                param.requires_grad = True
            layer.train()

    def forward(self, image, K):
        """
        Args:
            image (B,V,3,H,W)
            K (B,V,3,3)
        """
        B,V,_,image_h,image_w = image.shape
        image = image.view(-1,3,image_h,image_w)
        K = K.view(-1,3,3)
        out_hm,out_reg,feature = self.backbone(image)

        # BV,J,H,W
        heatmap = out_hm
        BV, J, H,W = heatmap.shape
        reg = out_reg.view(BV,J,2,H*W)
        
        # ind (BV,J)
        max_val, ind = torch.max(heatmap.view(BV,J, -1), dim=2)
        # ind_xy (BV,J,2)
        ind_y = ind // W  
        ind_x = ind % W
        ind_xy = torch.stack([ind_x, ind_y], dim=-1)

        # ind (BV,J,1,1) -> ind (BV,J,2,1) / reg (BV,J,2,HW) / reg_val (BV,J,2)
        reg_val = torch.gather(reg, 3, ind.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,2,-1)).squeeze(3)

        # coord (BV,J,2)
        coord = ind_xy.float() + reg_val
        coord[:, :, 0] = (coord[:, :, 0] / (W - 1)) * 2 - 1 
        coord[:, :, 1] = (coord[:, :, 1] / (H - 1)) * 2 - 1  
        coord = coord.unsqueeze(1)
        x0 = F.grid_sample(feature, coord, mode='bilinear', align_corners=True)
        x0 = x0.squeeze(2).transpose(1, 2) 

        # (BV,1,3,3)@(BV,J,3,1) -> (BV,J,3,1)
        x1 = (( torch.inverse(K[:,None]) @ eulid_to_homo(ind_xy.float())[...,None])).squeeze(-1)
        x1 = torch.cat([x1, max_val[...,None]], dim = -1)
        x1 = self.embed(x1)

        # (BV,J,256) -> (B,V,J*256)
        x = torch.cat([x0,x1], dim=-1).view(B,V,J*256)
        x = self.pose0(x) # (B,V,256)
        x = x.view(B,-1)
        out = self.pose1(x).view(B,-1,3)
        return out








