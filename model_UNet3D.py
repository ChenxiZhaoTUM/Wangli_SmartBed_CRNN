import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Convolution') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet3D(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(2, 4, 4), stride=(1, 1, 1),
                pad=(1, 1, 1), dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv3d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=factor, mode='trilinear'))
        block.add_module('%s_tconv' % name,
                         nn.Conv3d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm3d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout3d(dropout, inplace=True))

    return block


## ------------------------ UNet3D ---------------------- ##
class UNet3D(nn.Module):
    def __init__(self, channelExponent=3, dropout=0., time_step=10):
        super(UNet3D, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        ##### U-Net Encoder #####
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv3d(time_step, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet3D(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                  dropout=dropout, stride=2)

        self.layer3 = blockUNet3D(channels * 2, channels * 2, 'layer3', transposed=False, bn=True, relu=False,
                                  dropout=dropout, stride=2)

        self.layer4 = blockUNet3D(channels * 2, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                  dropout=dropout, size=(2, 5, 5), pad=(0, 1, 0))

        self.layer5 = blockUNet3D(channels * 4, channels * 8, 'layer5', transposed=False, bn=False, relu=False,
                                  dropout=dropout, size=(2, 2, 4), pad=0)

        ##### U-Net Decoder #####
        self.dlayer5 = blockUNet3D(channels * 8, channels * 4, 'dlayer5', transposed=True, bn=True, relu=True,
                                   dropout=dropout, size=(2, 3, 3), pad=(0, 1, 2))

        self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.dlayer4 = blockUNet3D(channels * 8, channels * 2, 'dlayer4', transposed=True, bn=True, relu=True,
                                   dropout=dropout, size=(4, 3, 3))

        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1))

        self.dlayer3 = blockUNet3D(channels * 4, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                   dropout=dropout, size=(4, 3, 3))

        self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1))

        self.dlayer2 = blockUNet3D(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                   dropout=dropout, size=(4, 3, 3))

        self.pool1 = nn.MaxPool3d(kernel_size=(4, 1, 1))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose3d(channels * 2, 1, (3, 4, 4), 2, 1, bias=True))

    def forward(self, x_allTimeStep):
        # x_allTimeStep: torch.Size([20, 10, 12, 32, 64])
        out1 = self.layer1(x_allTimeStep)  # torch.Size([20, 8, 6, 16, 32])
        out2 = self.layer2(out1)  # torch.Size([20, 16, 4, 8, 16])
        out3 = self.layer3(out2)  # torch.Size([20, 16, 3, 4, 8])
        out4 = self.layer4(out3)  # torch.Size([20, 32, 2, 2, 4])
        out5 = self.layer5(out4)  # torch.Size([20, 64, 1, 1, 1])

        dout5 = self.dlayer5(out5)  # torch.Size([20, 32, 1, 2, 4])
        out4_compression = self.pool4(out4)  # torch.Size([20, 32, 1, 2, 4])
        dout5_out4 = torch.cat([dout5, out4_compression], 1)  # torch.Size([20, 64, 1, 2, 4])

        dout4 = self.dlayer4(dout5_out4)  # torch.Size([20, 16, 1, 4, 8])
        out3_compression = self.pool3(out3)  # torch.Size([20, 16, 1, 4, 8])
        dout4_out3 = torch.cat([dout4, out3_compression], 1)  # torch.Size([20, 32, 1, 4, 8])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([20, 16, 1, 8, 16])
        out2_compression = self.pool2(out2)  # torch.Size([20, 16, 1, 8, 16])
        dout3_out2 = torch.cat([dout3, out2_compression], 1)  # torch.Size([20, 32, 1, 8, 16])

        dout2 = self.dlayer2(dout3_out2)  # torch.Size([20, 8, 1, 16, 32])
        out1_compression = self.pool1(out1)  # torch.Size([20, 8, 1, 16, 32])
        dout2_out1 = torch.cat([dout2, out1_compression], 1)  # torch.Size([20, 16, 1, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([20, 1, 1, 32, 64])

        dout1 = dout1.squeeze(1)

        return dout1


if __name__ == "__main__":
    input_tensor = torch.randn(20, 10, 12, 32, 64)  # (batch_size, time_step, channels, height, width)
    model = UNet3D()
    output_tensor = model(input_tensor)
    print(output_tensor.size())  # torch.Size([20, 1, 32, 64])
