import torch
import torch.nn as nn
# import torch.nn.functional as F


class ResAttrUnet3D2(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1):
        super(ResAttrUnet3D2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        def encode_kwargs_fn(level):
            num_stacks = max(level, 1)
            return {'num_stacks': num_stacks}

        down_features = [[30, 30], [60, 60], [120, 120], [240, 240], [320, 320]]
        up_features = [[320, 320], [240, 240], [120, 120], [60, 60], [30, 30]]
        bottom_features = [[320, 320]]

        paired_features = down_features + bottom_features + up_features

        self.net = Unet(in_channels=in_channels,
                        out_channels=out_channels,
                        paired_features=paired_features,
                        pool_block=ResBlock,
                        pool_kwargs={'stride': 2},
                        up_kwargs={'attention': True},
                        encode_block=ResBlockStack,
                        encode_kwargs_fn=encode_kwargs_fn,
                        decode_block=ResBlock)

    def forward(self, x):
        return self.net(x)


class ResAttrBNUnet3D(nn.Module):
    def __init__(self,
                 num_pool=4,
                 num_features=30,
                 in_channels=1,
                 out_channels=1):
        super(ResAttrBNUnet3D, self).__init__()
        self.num_pool = num_pool
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        def encode_kwargs_fn(level):
            num_stacks = max(level, 1)
            return {'num_stacks': num_stacks}

        paired_features = generate_paired_features(num_pool, num_features)

        self.net = Unet(in_channels=in_channels,
                        out_channels=out_channels,
                        paired_features=paired_features,
                        pool_block=ResBlock,
                        pool_kwargs={'stride': 2, 'norm_op': nn.BatchNorm3d},
                        up_kwargs={'attention': True, 'norm_op': nn.BatchNorm3d},
                        encode_block=ResBlockStack,
                        encode_kwargs={'norm_op': nn.BatchNorm3d},
                        encode_kwargs_fn=encode_kwargs_fn,
                        decode_block=ResBlock,
                        decode_kwargs={'norm_op': nn.BatchNorm3d})

    def forward(self, x):
        return self.net(x)


class ResAttrUnet3D(nn.Module):
    def __init__(self,
                 num_pool=4,
                 num_features=30,
                 in_channels=1,
                 out_channels=1):
        super(ResAttrUnet3D, self).__init__()
        self.num_pool = num_pool
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        def encode_kwargs_fn(level):
            num_stacks = max(level, 1)
            return {'num_stacks': num_stacks}

        paired_features = generate_paired_features(num_pool, num_features)

        self.net = Unet(in_channels=in_channels,
                        out_channels=out_channels,
                        paired_features=paired_features,
                        pool_block=ResBlock,
                        pool_kwargs={'stride': 2},
                        up_kwargs={'attention': True},
                        encode_block=ResBlockStack,
                        encode_kwargs_fn=encode_kwargs_fn,
                        decode_block=ResBlock)

    def forward(self, x):
        return self.net(x)


class ResUnet3D(nn.Module):
    def __init__(self,
                 num_pool=4,
                 num_features=30,
                 in_channels=1,
                 out_channels=1):
        super(ResUnet3D, self).__init__()
        self.num_pool = num_pool
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        def encode_kwargs_fn(level):
            num_stacks = max(level, 1)
            return {'num_stacks': num_stacks}

        paired_features = generate_paired_features(num_pool, num_features)

        self.net = Unet(in_channels=in_channels,
                        out_channels=out_channels,
                        paired_features=paired_features,
                        pool_block=ResBlock,
                        pool_kwargs={'stride': 2},
                        encode_block=ResBlockStack,
                        encode_kwargs_fn=encode_kwargs_fn,
                        decode_block=ResBlock)

    def forward(self, x):
        return self.net(x)


def generate_paired_features(num_pool, num_features):
    down_features = [[num_features*(2**i), num_features*(2**i)]
                     for i in range(num_pool)]
    up_features = [[num_features*(2**i), num_features*(2**i)]
                   for i in range(num_pool-1, -1, -1)]
    bottom_features = [[num_features*(2**num_pool), num_features*(2**num_pool)]]
    return down_features+bottom_features+up_features


def generate_paired_features2(num_pool, num_features):
    down_features = [[num_features*(2**i), num_features*(2**(i+1))]
                     for i in range(num_pool)]
    up_features = [[num_features*(2**i), num_features*(2**i)]
                   for i in range(num_pool-1, -1, -1)]
    bottom_features = [[num_features*(2**num_pool), num_features*(2**num_pool)]]
    return down_features+bottom_features+up_features


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = conv_op(in_channels, out_channels, **conv_kwargs)
        if dropout_op:
            self.dropout = dropout_op(**dropout_kwargs)
        else:
            self.dropout = None
        self.norm = norm_op(out_channels, **norm_kwargs)
        self.nonlin = nonlin_op(**nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return self.nonlin(self.norm(x))


class ConvBlockStack(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_stacks=2,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ConvBlockStack, self).__init__()

        self.conv_blocks = nn.ModuleList(
            [ConvBlock(in_channels=in_channels if i == 0 else out_channels,
                       out_channels=out_channels,
                       conv_op=conv_op, conv_kwargs=conv_kwargs,
                       dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
                       norm_op=norm_op, norm_kwargs=norm_kwargs,
                       nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs
                       )for i in range(num_stacks)]
        )

    def forward(self, x):
        for conv in self.conv_blocks:
            x = conv(x)

        return x


class RecBlock(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self,
                 out_channels, t=2,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(RecBlock, self).__init__()

        self.t = t
        self.out_channels = out_channels
        self.conv = ConvBlock(out_channels, out_channels,
                              conv_op=conv_op, conv_kwargs=conv_kwargs,
                              dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
                              norm_op=norm_op, norm_kwargs=norm_kwargs,
                              nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs
                              )

    def forward(self, x):
        out = self.conv(x)
        for i in range(self.t):
            out = self.conv(out + x)
        return out


class ResRecBlock(nn.Module):
    """
    Recurrent Residual Convolutional Block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 t=2,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ResRecBlock, self).__init__()

        self.t = t
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.rcnn = nn.Sequential(
            RecBlock(out_channels, t=t,
                     conv_op=conv_op, conv_kwargs=conv_kwargs,
                     dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
                     norm_op=norm_op, norm_kwargs=norm_kwargs,
                     nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs
                     ),
            RecBlock(out_channels, t=t,
                     conv_op=conv_op, conv_kwargs=conv_kwargs,
                     dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
                     norm_op=norm_op, norm_kwargs=norm_kwargs,
                     nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs
                     )
        )
        self.conv = conv_op(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            skip = self.conv(x)
        else:
            skip = x
        x = self.rcnn(skip)
        return x + skip


class ConvTrans3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ConvTrans3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1),
            nn.ConstantPad3d(padding=(0, 1, 0, 1, 0, 1), value=0),
            norm_op(out_channels, **norm_kwargs),
            nonlin_op(**nonlin_kwargs)
        )

    def forward(self, x):
        return self.up(x)


class UpConcat(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_trans_op=ConvTrans3D,
                 attention=False,
                 att_conv_op=nn.Conv3d,
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(UpConcat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = attention
        self.conv_trans = conv_trans_op(in_channels, out_channels,
                                        norm_op=norm_op, norm_kwargs=norm_kwargs,
                                        nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs)
        if attention:
            self.att_gate = AttBlock(out_channels, conv_op=att_conv_op,
                                     nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs)

    def forward(self, x, skip):
        x = self.conv_trans(x)
        if self.attention:
            skip = self.att_gate(skip, x)
        return torch.cat((x, skip), dim=1)


class AttBlock(nn.Module):

    def __init__(self,
                 out_channels,
                 conv_op=nn.Conv3d,
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(AttBlock, self).__init__()

        self.conv = conv_op(out_channels, out_channels, kernel_size=1)
        self.lrelu = nonlin_op(**nonlin_kwargs)
        self.active = nn.Sigmoid()

    def forward(self, x, gate):
        x = self.conv(x)
        g = self.conv(gate)
        f = self.lrelu(x+g)
        rate = self.active(self.conv(f))
        return x * rate


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = conv_op(in_channels, out_channels, stride=stride, **conv_kwargs)
        self.conv2 = conv_op(out_channels, out_channels, **conv_kwargs)

        if dropout_op:
            self.dropout = dropout_op(**dropout_kwargs)
        else:
            self.dropout = None
        self.norm = norm_op(out_channels, **norm_kwargs)
        self.nonlin = nonlin_op(**nonlin_kwargs)
        self.skip_conv = conv_op(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels or self.stride != 1:
            skip = self.skip_conv(x)
        else:
            skip = x

        x = self.conv1(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.nonlin(self.norm(x))
        x = self.conv2(x)
        return self.nonlin(self.norm(x)+skip)


class ResBlockStack(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 num_stacks=2,
                 conv_op=nn.Conv3d,
                 conv_kwargs={'kernel_size': 3, 'padding': 1},
                 dropout_op=nn.Dropout3d,
                 dropout_kwargs={'p': 0.5, 'inplace': True},
                 norm_op=nn.InstanceNorm3d,
                 norm_kwargs={},
                 nonlin_op=nn.LeakyReLU,
                 nonlin_kwargs={'inplace': True}):
        super(ResBlockStack, self).__init__()

        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels=in_channels if i == 0 else out_channels,
                      out_channels=out_channels,
                      stride=stride if i == 0 else 1,
                      conv_op=conv_op, conv_kwargs=conv_kwargs,
                      dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
                      norm_op=norm_op, norm_kwargs=norm_kwargs,
                      nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs)
             for i in range(num_stacks)]
        )

    def forward(self, x):
        for res in self.res_blocks:
            x = res(x)
        return x


class MaxPoolBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_op=nn.MaxPool3d,
                 pool_kwargs={'kernel_size': 2, 'stride': 2}):
        super(MaxPoolBlock, self).__init__()

        self.pool = pool_op(**pool_kwargs)

    def forward(self, x):
        return self.pool(x)


def none_fn(level):
    return {}


class Unet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 paired_features,
                 pool_block=MaxPoolBlock,
                 pool_kwargs={},
                 pool_kwargs_fn=none_fn,
                 up_block=UpConcat,
                 up_kwargs={},
                 up_kwargs_fn=none_fn,
                 encode_block=ConvBlockStack,
                 encode_kwargs={},
                 encode_kwargs_fn=none_fn,
                 decode_block=ConvBlockStack,
                 decode_kwargs={},
                 decode_kwargs_fn=none_fn,
                 conv_op=nn.Conv3d):
        super(Unet, self).__init__()

        num_pairs = len(paired_features)
        assert (num_pairs % 2) == 1, 'Number of paired features must be odd number.'
        self.num_pool = num_pairs // 2
        assert self.num_pool > 0, 'At least one pool.'
        pool_blocks_list = []
        up_blocks_list = []
        encode_blocks_list = []
        decode_blocks_list = []

        for i in range(self.num_pool):
            pool_kwargs_fns = pool_kwargs_fn(i)
            up_kwargs_fns = up_kwargs_fn(i)
            encode_kwargs_fns = encode_kwargs_fn(i)
            decode_kwargs_fns = decode_kwargs_fn(i)

            pool_blocks_list.append(
                pool_block(paired_features[i][1],
                           paired_features[i+1][0],
                           ** pool_kwargs_fns,
                           ** pool_kwargs))

            up_blocks_list.append(
                up_block(paired_features[num_pairs-i-2][1],
                         paired_features[num_pairs-i-1][0],
                         ** up_kwargs_fns,
                         ** up_kwargs))

            encode_blocks_list.append(
                encode_block(paired_features[i][0],
                             paired_features[i][1],
                             ** encode_kwargs_fns,
                             ** encode_kwargs))

            decode_blocks_list.append(
                decode_block(paired_features[num_pairs-i-1][0]+paired_features[i][1],
                             paired_features[num_pairs-i-1][1],
                             ** decode_kwargs_fns,
                             ** decode_kwargs))

        encode_kwargs_fns = encode_kwargs_fn(self.num_pool)
        encode_blocks_list.append(
            encode_block(paired_features[self.num_pool][0],
                         paired_features[self.num_pool][1],
                         ** encode_kwargs_fns,
                         ** encode_kwargs))

        self.pool_blocks = nn.ModuleList(pool_blocks_list)
        self.up_blocks = nn.ModuleList(up_blocks_list)
        self.encode_blocks = nn.ModuleList(encode_blocks_list)
        self.decode_blocks = nn.ModuleList(decode_blocks_list)

        self.conv = conv_op(in_channels,
                            paired_features[0][0],
                            kernel_size=3,
                            padding=1)
        self.fc = conv_op(paired_features[num_pairs-1][1],
                          out_channels,
                          kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        skip_layers = []
        for i in range(self.num_pool):
            x = self.encode_blocks[i](x)
            skip_layers.append(x)
            x = self.pool_blocks[i](x)

        x = self.encode_blocks[-1](x)

        for i in range(self.num_pool-1, -1, -1):
            x = self.up_blocks[i](x, skip_layers[i])
            x = self.decode_blocks[i](x)

        x = self.fc(x)

        return x


# class Unet2(nn.Module):
#     def __init__(self, in_channels, out_channels, num_pool,
#                  pool_block, up_block, encode_block, decode_block, conv_op=nn.Conv3d):
#         super(Unet2, self).__init__()
#         self.num_pool = num_pool
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.pool_blocks = nn.ModuleList(
#             [pool_block.make(i) for i in range(num_pool)]
#         )
#         self.up_block = nn.ModuleList(
#             [up_block.make(i) for i in range(num_pool)]
#         )

#         self.encode_blocks = nn.ModuleList(
#             [encode_block.make(i) for i in range(num_pool+1)]
#         )
#         self.decode_blocks = nn.ModuleList(
#             [decode_block.make(i) for i in range(num_pool)]
#         )

#         self.conv = conv_op(in_channels, encode_block.in_channels_list[0],
#                               kernel_size=3, padding=1)
#         self.fc = conv_op(decode_block.out_channels_list[0], out_channels, kernel_size=1)
#         self.active = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv(x)
#         skip_layers = []
#         for i in range(self.num_pool):
#             x = self.encode_blocks[i](x)
#             skip_layers.append(x)
#             x = self.pool_blocks[i](x)

#         x = self.encode_blocks[-1](x)

#         for i in range(self.num_pool-1, -1, -1):
#             x = self.up_block[i](x, skip_layers[i])
#             x = self.decode_blocks[i](x)

#         x = self.fc(x)
#         x = self.active(x)

#         return x

# class BlockMaker():
#     def __init__(self, in_channels_list, out_channels_list,
#                  conv_op=nn.Conv3d, conv_kwargs={'kernel_size': 3, 'padding': 1},
#                  dropout_op=nn.Dropout3d, dropout_kwargs={'p': 0.5, 'inplace': True},
#                  norm_op=nn.InstanceNorm3d, norm_kwargs={},
#                  nonlin_op=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):

#         self.conv_op = conv_op
#         self.conv_kwargs = conv_kwargs

#         self.dropout_op = dropout_op
#         self.dropout_kwargs = dropout_kwargs

#         self.norm_op = norm_op
#         self.norm_kwargs = norm_kwargs

#         self.nonlin_op = nonlin_op
#         self.nonlin_kwargs = nonlin_kwargs

#         self.in_channels_list = in_channels_list
#         self.out_channels_list = out_channels_list

#     def make(self, index):
#         return ConvBlock(self.in_channels_list[index], self.out_channels_list[index],
#                          conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
#                          dropout_op=self.dropout_op, dropout_kwargs=self.dropout_kwargs,
#                          norm_op=self.norm_op, norm_kwargs=self.norm_kwargs,
#                          nonlin_op=self.nonlin_op, nonlin_kwargs=self.nonlin_kwargs)


# class ResBlockMaker(BlockMaker):
#     def __init__(self, in_channels_list, out_channels_list, stride=1,
#                  conv_op=nn.Conv3d, conv_kwargs={'kernel_size': 3, 'padding': 1},
#                  dropout_op=nn.Dropout3d, dropout_kwargs={'p': 0.5, 'inplace': True},
#                  norm_op=nn.InstanceNorm3d, norm_kwargs={},
#                  nonlin_op=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):

#         super(ResBlockMaker, self).__init__(
#             in_channels_list, out_channels_list,
#             conv_op=conv_op, conv_kwargs=conv_kwargs,
#             dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
#             norm_op=norm_op, norm_kwargs=norm_kwargs,
#             nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs)

#         self.stride = stride

#     def make(self, index):
#         return ResBlock(self.in_channels_list[index], self.out_channels_list[index],
#                         stride=self.stride,
#                         conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
#                         dropout_op=self.dropout_op, dropout_kwargs=self.dropout_kwargs,
#                         norm_op=self.norm_op, norm_kwargs=self.norm_kwargs,
#                         nonlin_op=self.nonlin_op, nonlin_kwargs=self.nonlin_kwargs)


# class ResBlockStackMaker(ResBlockMaker):
#     def __init__(self, in_channels_list, out_channels_list, stride=1,
#                  conv_op=nn.Conv3d, conv_kwargs={'kernel_size': 3, 'padding': 1},
#                  dropout_op=nn.Dropout3d, dropout_kwargs={'p': 0.5, 'inplace': True},
#                  norm_op=nn.InstanceNorm3d, norm_kwargs={},
#                  nonlin_op=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):
#         super(ResBlockStackMaker, self).__init__(
#             in_channels_list, out_channels_list, stride=stride,
#             conv_op=conv_op, conv_kwargs=conv_kwargs,
#             dropout_op=dropout_op, dropout_kwargs=dropout_kwargs,
#             norm_op=norm_op, norm_kwargs=norm_kwargs,
#             nonlin_op=nonlin_op, nonlin_kwargs=nonlin_kwargs)

#     def make(self, index):
#         num_stacks = max(index, 1)
#         return ResBlockStack(self.in_channels_list[index], self.out_channels_list[index],
#                              stride=self.stride, num_stacks=num_stacks,
#                              conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
#                              dropout_op=self.dropout_op, dropout_kwargs=self.dropout_kwargs,
#                              norm_op=self.norm_op, norm_kwargs=self.norm_kwargs,
#                              nonlin_op=self.nonlin_op, nonlin_kwargs=self.nonlin_kwargs)


# class UpBlockMaker():
#     def __init__(self, in_channels_list, out_channels_list,
#                  conv_trans_op=ConvTrans3D, attention=False, att_conv_op=nn.Conv3d,
#                  norm_op=nn.InstanceNorm3d, norm_kwargs={},
#                  nonlin_op=nn.LeakyReLU, nonlin_kwargs={'inplace': True}):

#         self.attention = attention
#         self.conv_trans_op = conv_trans_op
#         self.att_conv_op = att_conv_op
#         self.norm_op = norm_op
#         self.norm_kwargs = norm_kwargs
#         self.nonlin_op = nonlin_op
#         self.nonlin_kwargs = nonlin_kwargs
#         self.in_channels_list = in_channels_list
#         self.out_channels_list = out_channels_list

#     def make(self, index):
#         return UpConcat(self.in_channels_list[index], self.out_channels_list[index],
#                         conv_trans_op=self.conv_trans_op,
#                         attention=self.attention, att_conv_op=self.att_conv_op,
#                         norm_op=self.norm_op, norm_kwargs=self.norm_kwargs,
#                         nonlin_op=self.nonlin_op, nonlin_kwargs=self.nonlin_kwargs)
