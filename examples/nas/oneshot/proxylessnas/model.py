import torch
import nni.retiarii.nn.pytorch as nn
import math

import ops
import putils
from nni.retiarii.nn.pytorch import LayerChoice 
### Changjae Lee @ 2022-09-17 
import nni 
### Changjae Lee @ 2022-09-22 
from nni.retiarii.serializer import model_wrapper 
from blocks import ShuffleNetBlock, ShuffleXceptionBlock 

class SearchMobileNet(nn.Module):
    def __init__(self,
                 width_stages=[24,40,80,96,192,320],
                 n_cell_stages=[4,4,4,4,4,1],
                 stride_stages=[2,2,2,1,2,1],
                 width_mult=1, n_classes=1000,
                 dropout_rate=0, bn_param=(0.1, 1e-3)):
        """
        Parameters
        ----------
        width_stages: str
            width (output channels) of each cell stage in the block
        n_cell_stages: str
            number of cells in each cell stage
        stride_strages: str
            stride of each cell stage in the block
        width_mult : int
            the scale factor of width
        """
        super(SearchMobileNet, self).__init__()

        input_channel = putils.make_divisible(32 * width_mult, 8)
        first_cell_width = putils.make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = putils.make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        first_conv = ops.ConvLayer(3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = ops.OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                op_candidates = [ops.OPS['3x3_MBConv3'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv6'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv6'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv6'](input_channel, width, stride)]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [ops.OPS['Zero'](input_channel, width, stride)]
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                else:
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = ops.IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = ops.MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        last_channel = putils.make_devisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ops.ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = ops.LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

### Changjae Lee @ 2022-09-21 
# X -> output_size=16 
class SearchTinyMLNet(nn.Module):
    def __init__(self,
                 width_stages=[8,8,16,16,24,40],
                 n_cell_stages=[4,4,4,4,4,1],
                 stride_stages=[1,1,1,1,2,1],
                 width_mult=1, n_classes=2,
                 dropout_rate=0, bn_param=(0.1, 1e-3), 
                 output_size=16):
        """
        Parameters
        ----------
        width_stages: str
            width (output channels) of each cell stage in the block
        n_cell_stages: str
            number of cells in each cell stage
        stride_strages: str
            stride of each cell stage in the block
        width_mult : int
            the scale factor of width
        """
        ### Changjae Lee @ 2022-09-16 
        super(SearchTinyMLNet, self).__init__()

        ### Changjae Lee @ 2022-09-17 
        self.conv1 = nni.trace(torch.nn.Conv1d)(1, 3, kernel_size=6, stride=1, padding=0, dilation=5)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, padding=0)
        
        input_channel = putils.make_divisible(32 * width_mult, 8)
        first_cell_width = putils.make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = putils.make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        ### Changjae Lee @ 2022-09-16 
        # stride=2 -> stride=1
        first_conv = ops.ConvLayer(3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = ops.OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                op_candidates = [ops.OPS['3x3_MBConv3'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv6'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv6'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv6'](input_channel, width, stride)]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [ops.OPS['Zero'](input_channel, width, stride)]
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                else:
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = ops.IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = ops.MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        ### Changjae Lee @ 2022-09-17 
        # 1280 -> 64 
        last_channel = putils.make_devisible(output_size * width_mult, 8) if width_mult > 1.0 else output_size
        feature_mix_layer = ops.ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = ops.LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        ### Changjae Lee @ 2022-09-17 
        x = self.conv1(x.view(-1, 1, x.shape[-1])) 
        x = self.conv2(x.view(-1, 3, 35, 35))
        
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

### Changjae Lee @ 2022-09-21 
# X -> div=2 
class SearchTinyMLNet_div(nn.Module):
    def __init__(self,
                 width_stages=[8,8,16,16,24,40],
                 n_cell_stages=[4,4,4,4,4,1],
                 stride_stages=[1,1,1,1,2,1],
                 width_mult=1, n_classes=2,
                 dropout_rate=0, bn_param=(0.1, 1e-3), 
                 output_size=16, div=2):
        """
        Parameters
        ----------
        width_stages: str
            width (output channels) of each cell stage in the block
        n_cell_stages: str
            number of cells in each cell stage
        stride_strages: str
            stride of each cell stage in the block
        width_mult : int
            the scale factor of width
        """
        ### Changjae Lee @ 2022-09-21 
        super(SearchTinyMLNet_div, self).__init__()

        ### Changjae Lee @ 2022-09-21 
        #div = 2 

        ### Changjae Lee @ 2022-09-17 
        self.conv1 = nni.trace(torch.nn.Conv1d)(1, 3, kernel_size=6, stride=1, padding=0, dilation=5)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, padding=0)
        
        ### Changjae Lee @ 2022-09-21 
        # 8 -> div 
        input_channel = putils.make_divisible(32 * width_mult, div)
        first_cell_width = putils.make_divisible(16 * width_mult, div)
        for i in range(len(width_stages)):
            width_stages[i] = putils.make_divisible(width_stages[i] * width_mult, div)
        # first conv
        ### Changjae Lee @ 2022-09-16 
        # stride=2 -> stride=1
        first_conv = ops.ConvLayer(3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = ops.OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                op_candidates = [ops.OPS['3x3_MBConv3'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv6'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv6'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv6'](input_channel, width, stride)]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [ops.OPS['Zero'](input_channel, width, stride)]
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                else:
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = ops.IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = ops.MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        ### Changjae Lee @ 2022-09-17 
        # 1280 -> 64 
        ### Changjae Lee @ 2022-09-21  
        # 8 -> div 
        last_channel = putils.make_devisible(output_size * width_mult, div) if width_mult > 1.0 else output_size
        feature_mix_layer = ops.ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = ops.LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        ### Changjae Lee @ 2022-09-17 
        x = self.conv1(x.view(-1, 1, x.shape[-1])) 
        x = self.conv2(x.view(-1, 3, 35, 35))
        
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

### Changjae Lee @ 2022-09-22 
# X -> div=2 
class SearchTinyMLNet_e(nn.Module):
    def __init__(self,
                 width_stages=[8,8,16,16,24,40],
                 n_cell_stages=[4,4,4,4,4,1],
                 stride_stages=[1,1,1,1,2,1],
                 width_mult=1, n_classes=2,
                 dropout_rate=0, bn_param=(0.1, 1e-3), 
                 output_size=16, div=2):
        """
        Parameters
        ----------
        width_stages: str
            width (output channels) of each cell stage in the block
        n_cell_stages: str
            number of cells in each cell stage
        stride_strages: str
            stride of each cell stage in the block
        width_mult : int
            the scale factor of width
        """
        ### Changjae Lee @ 2022-09-22 
        super(SearchTinyMLNet_e, self).__init__()

        ### Changjae Lee @ 2022-09-21 
        #div = 2 

        ### Changjae Lee @ 2022-09-17 
        self.conv1 = nni.trace(torch.nn.Conv1d)(1, 3, kernel_size=6, stride=1, padding=0, dilation=5)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, padding=0)
        
        ### Changjae Lee @ 2022-09-21 
        # 8 -> div 
        input_channel = putils.make_divisible(32 * width_mult, div)
        first_cell_width = putils.make_divisible(16 * width_mult, div)
        for i in range(len(width_stages)):
            width_stages[i] = putils.make_divisible(width_stages[i] * width_mult, div)
        # first conv
        ### Changjae Lee @ 2022-09-16 
        # stride=2 -> stride=1
        first_conv = ops.ConvLayer(3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = ops.OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                ### Changjae Lee @ 2022-09-22 
                op_candidates = [ops.OPS['3x3_MBConv2'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv3'](input_channel, width, stride),
                                 ops.OPS['3x3_MBConv4'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv2'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                                 ops.OPS['5x5_MBConv4'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv2'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                                 ops.OPS['7x7_MBConv4'](input_channel, width, stride)]
                # op_candidates = [ops.OPS['3x3_MBConv1'](input_channel, width, stride), 
                #                  ops.OPS['3x3_MBConv2'](input_channel, width, stride),
                #                  ops.OPS['3x3_MBConv4'](input_channel, width, stride),
                #                  ops.OPS['3x3_MBConv6'](input_channel, width, stride),
                #                  ops.OPS['5x5_MBConv3'](input_channel, width, stride),
                #                  ops.OPS['5x5_MBConv6'](input_channel, width, stride),
                #                  ops.OPS['7x7_MBConv3'](input_channel, width, stride),
                #                  ops.OPS['7x7_MBConv6'](input_channel, width, stride)]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [ops.OPS['Zero'](input_channel, width, stride)]
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                else:
                    conv_op = LayerChoice(op_candidates, label="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = ops.IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = ops.MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        ### Changjae Lee @ 2022-09-17 
        # 1280 -> 64 
        ### Changjae Lee @ 2022-09-21  
        # 8 -> div 
        last_channel = putils.make_devisible(output_size * width_mult, div) if width_mult > 1.0 else output_size
        feature_mix_layer = ops.ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = ops.LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, x):
        ### Changjae Lee @ 2022-09-17 
        x = self.conv1(x.view(-1, 1, x.shape[-1])) 
        x = self.conv2(x.view(-1, 3, 35, 35))
        
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

### Changjae Lee @ 2022-09-22 
@model_wrapper
class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    ### Changjae Lee @ 2022-09-22 
    # input_size, last_conv_channels, n_classes 
    # stage_blocks, stage_channels 
    # stage_strides, pool 
    def __init__(self, input_size=32, first_conv_channels=16, last_conv_channels=16,
                 n_classes=2, affine=False, 
                 stage_blocks=[4, 4, 8, 4], stage_channels=[64, 160, 320, 640], 
                 stage_strides=[1, 1, 2, 1], pool='a'):
        ### Changjae Lee @ 2022-09-22 
        super().__init__()

        assert input_size % 32 == 0
        ### Changjae Lee @ 2022-09-22 
        assert len(stage_blocks) == len(stage_channels) 
        self.stage_blocks = stage_blocks 
        self.stage_channels = stage_channels 
        self.stage_strides = stage_strides 
        self.pool = pool 
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes
        self._affine = affine
        self._layerchoice_count = 0

        ### Changjae Lee @ 2022-09-22  
        self.conv1 = nni.trace(torch.nn.Conv1d)(1, 3, kernel_size=6, stride=1, padding=0, dilation=5)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, padding=0) 

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self._feature_map_size //= 2        # 16 

        p_channels = first_conv_channels
        features = []
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        ### Changjae Lee @ 2022-09-22 
        self.globalpool = nn.AvgPool2d(self._feature_map_size) if self.pool == 'a' else nn.MaxPool2d(self._feature_map_size) 
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    ### Changjae Lee @ 2022-09-22 
    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            ### Changjae Lee @ 2022-09-22 
            stride = 2 if i == 0 else 1 
            #stride = strides[i] 
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            self._layerchoice_count += 1
            choice_block = LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride, affine=self._affine),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride, affine=self._affine)
            ], label="LayerChoice" + str(self._layerchoice_count))
            result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)

        ### Changjae Lee @ 2022-09-22 
        x = self.conv1(x.view(-1, 1, x.shape[-1])) 
        x = self.conv2(x.view(-1, 3, 35, 35)) 

        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                else:
                    torch.nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


def load_and_parse_state_dict(filepath="./data/checkpoint-150000.pth.tar"):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    result = dict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[len("module."):]
        result[k] = v
    return result