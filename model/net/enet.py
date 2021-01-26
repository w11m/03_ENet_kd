import torch.nn as nn
import torch

import time
class InitialBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):

    def __init__(self, num_classes,CHset, encoder_relu=False, decoder_relu=True,reduce2stage=False,reduce3stage=False, stage3 = True ):
        super().__init__()
        self.reduce2stage = reduce2stage
        self.reduce3stage = reduce3stage
        self.stage3 = stage3

        self.initial_block = InitialBlock(3, CHset[0], relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(CHset[0], CHset[1],
                                                    return_indices=True,
                                                    dropout_prob=0.01,
                                                    relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(CHset[1], CHset[2],
                                                    return_indices=True,
                                                    dropout_prob=0.1,
                                                    relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(CHset[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(CHset[2], dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(CHset[2],
                                               kernel_size=5,
                                               padding=2,
                                               asymmetric=True,
                                               dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(CHset[2], dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        if self.reduce2stage:
            pass
        else:
            self.regular2_5 = RegularBottleneck(CHset[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated2_6 = RegularBottleneck(CHset[2], dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric2_7 = RegularBottleneck(CHset[2],
                                                   kernel_size=5,
                                                   asymmetric=True,
                                                   padding=2,
                                                   dropout_prob=0.1,
                                                   relu=encoder_relu)
            self.dilated2_8 = RegularBottleneck(CHset[2], dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        if self.reduce3stage and self.stage3:
            self.regular3_0 = RegularBottleneck(CHset[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_1 = RegularBottleneck(CHset[2], dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric3_2 = RegularBottleneck(CHset[2],
                                                   kernel_size=5,
                                                   padding=2,
                                                   asymmetric=True,
                                                   dropout_prob=0.1,
                                                   relu=encoder_relu)
            self.dilated3_3 = RegularBottleneck(CHset[2], dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        elif not self.stage3:
            pass
        else:
            self.regular3_0 = RegularBottleneck(CHset[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_1 = RegularBottleneck(CHset[2], dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric3_2 = RegularBottleneck(CHset[2],
                                                   kernel_size=5,
                                                   padding=2,
                                                   asymmetric=True,
                                                   dropout_prob=0.1,
                                                   relu=encoder_relu)
            self.dilated3_3 = RegularBottleneck(CHset[2], dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
            self.regular3_4 = RegularBottleneck(CHset[2], padding=1, dropout_prob=0.1, relu=encoder_relu)
            self.dilated3_5 = RegularBottleneck(CHset[2], dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
            self.asymmetric3_6 = RegularBottleneck(CHset[2],
                                                   kernel_size=5,
                                                   asymmetric=True,
                                                   padding=2,
                                                   dropout_prob=0.1,
                                                   relu=encoder_relu)
            self.dilated3_7 = RegularBottleneck(CHset[2], dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(CHset[2], CHset[1], dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(CHset[1], padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(CHset[1], CHset[0], dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(CHset[0], padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(CHset[0],
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    # def forward(self, x):
    #     # Initial block
    #
    #     input_size = x.size()
    #
    #     batch = input_size[0]
    #     w, h = input_size[2], input_size[3]
    #     out_size = ([batch, 3, w * 2, h * 2])
    #     start = time.time()
    #     t1 = time.time()
    #     x = self.initial_block(x)
    #     t2= time.time()
    #     print("initial_block :{}".format(t2-t1))
    #     # Stage 1 - Encoder
    #     stage1_input_size = x.size()
    #     t1 = time.time()
    #     x, max_indices1_0 = self.downsample1_0(x)
    #     t2 = time.time()
    #     print("downsample1_0 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular1_1(x)
    #     t2 = time.time()
    #     print("regular1_1 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular1_2(x)
    #     t2 = time.time()
    #     print("regular1_2 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular1_3(x)
    #     t2 = time.time()
    #     print("regular1_3 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular1_4(x)
    #     t2 = time.time()
    #     print("regular1_4 :{}".format(t2 - t1))
    #
    #     # Stage 2 - Encoder
    #     stage2_input_size = x.size()
    #     t1 = time.time()
    #     x, max_indices2_0 = self.downsample2_0(x)
    #     t2= time.time()
    #     print("downsample2_0 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular2_1(x)
    #     t2 = time.time()
    #     print("regular2_1 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.dilated2_2(x)
    #     t2 = time.time()
    #     print("dilated2_2 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.asymmetric2_3(x)
    #     t2 = time.time()
    #     print("asymmetric2_3 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.dilated2_4(x)
    #     t2 = time.time()
    #     print("dilated2_4 :{}".format(t2 - t1))
    #
    #     if self.reduce2stage:
    #         pass
    #     else:
    #         t1 = time.time()
    #         x = self.regular2_5(x)
    #         t2 = time.time()
    #         print("regular2_5 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated2_6(x)
    #         t2 = time.time()
    #         print("dilated2_6 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.asymmetric2_7(x)
    #         t2 = time.time()
    #         print("asymmetric2_7 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated2_8(x)
    #         t2 = time.time()
    #         print("dilated2_8 :{}".format(t2 - t1))
    #
    #     # Stage 3 - Encoder
    #     if self.reduce3stage and  self.stage3:
    #         x = self.regular3_0(x)
    #         x = self.dilated3_1(x)
    #         x = self.asymmetric3_2(x)
    #         x = self.dilated3_3(x)
    #     elif not self.stage3 :
    #         pass
    #     else:
    #         t1 = time.time()
    #         x = self.regular3_0(x)
    #         t2 = time.time()
    #         print("regular3_0 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated3_1(x)
    #         t2 = time.time()
    #         print("dilated3_1 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.asymmetric3_2(x)
    #         t2 = time.time()
    #         print("asymmetric3_2 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated3_3(x)
    #         t2 = time.time()
    #         print("dilated3_3 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.regular3_4(x)
    #         t2 = time.time()
    #         print("regular3_4 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated3_5(x)
    #         t2 = time.time()
    #         print("dilated3_5 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.asymmetric3_6(x)
    #         t2 = time.time()
    #         print("asymmetric3_6 :{}".format(t2 - t1))
    #
    #         t1 = time.time()
    #         x = self.dilated3_7(x)
    #         t2 = time.time()
    #         print("dilated3_7 :{}".format(t2 - t1))
    #
    #     self.feature1 = x.detach()
    #     # Stage 4 - Decoder
    #     t1 = time.time()
    #     x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
    #     t2 = time.time()
    #     print("upsample4_0 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular4_1(x)
    #     t2 = time.time()
    #     print("regular4_1 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular4_2(x)
    #     t2 = time.time()
    #     print("regular4_2 :{}".format(t2 - t1))
    #     # Stage 5 - Decoder
    #     t1 = time.time()
    #     x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
    #     t2 = time.time()
    #     print("upsample5_0 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.regular5_1(x)
    #     t2 = time.time()
    #     print("regular5_1 :{}".format(t2 - t1))
    #
    #     t1 = time.time()
    #     x = self.transposed_conv(x, output_size=input_size)
    #     t2 = time.time()
    #     end = time.time()
    #     print("transposed_conv :{}".format(t2 - t1))
    #     print("total_time :{}\n".format(end - start))
    #     return x

    def forward(self, x):
        # Initial block

        input_size = x.size()

        batch = input_size[0]
        w, h = input_size[2], input_size[3]
        out_size = ([batch, 3, w * 2, h * 2])
        x = self.initial_block(x)
        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)

        if self.reduce2stage:
            pass
        else:

            x = self.regular2_5(x)
            x = self.dilated2_6(x)
            x = self.asymmetric2_7(x)
            x = self.dilated2_8(x)

        # Stage 3 - Encoder
        if self.reduce3stage and self.stage3:
            x = self.regular3_0(x)
            x = self.dilated3_1(x)
            x = self.asymmetric3_2(x)
            x = self.dilated3_3(x)
        elif not self.stage3:
            pass
        else:

            x = self.regular3_0(x)
            x = self.dilated3_1(x)
            x = self.asymmetric3_2(x)
            x = self.dilated3_3(x)
            x = self.regular3_4(x)
            x = self.dilated3_5(x)
            x = self.asymmetric3_6(x)
            x = self.dilated3_7(x)

        self.feature1 = x.detach()
        # Stage 4 - Decoder

        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)
        return x

if __name__ == '__main__':
    model = ENet('ENet',19)
    # model = ENet_model('ENet_slim1', 19)
    # model = ENet_model('ENet_slim2', 19)
    # model = ENet_model('ENet_slim3', 19)
    fake_input= torch.rand(1,3,512,512)
    predict = model(fake_input)