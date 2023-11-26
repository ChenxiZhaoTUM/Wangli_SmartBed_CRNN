import torch
import torch.nn as nn
from ConvLSTM_pytorch import ConvLSTM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Convolution') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(4, 4), stride=(1, 1), pad=(1, 1),
              dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=factor, mode='bilinear'))
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block


## ------------------------ CnnEncoder * 10 + CnnDecoder * 10 + ConvLSTM ---------------------- ##
class UNet_ConvLSTM(nn.Module):
    def __init__(self, channelExponent=3, dropout=0.):
        super(UNet_ConvLSTM, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(12, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 2, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer4 = blockUNet(channels * 2, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer5 = blockUNet(channels * 4, channels * 8, 'layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(2, 3), pad=0)

        self.dlayer5 = blockUNet(channels * 8, channels * 4, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer4 = blockUNet(channels * 8, channels * 2, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer3 = blockUNet(channels * 4, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 1, 4, 2, 1, bias=True))

        self.convlstm = ConvLSTM(1, 1, (3, 3), 1, True, True, False, True)
        # input_channel, output_channel, kernel_size, num_layers, batch_first, bias, return_all_layers, only_last_time

    def forward(self, x_3d):
        unet_embed_seq = []
        for t in range(x_3d.size(1)):
            out1 = self.layer1(x_3d[:, t, :, :, :])
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)

            dout5 = self.dlayer5(out5)
            dout5_out4 = torch.cat([dout5, out4], 1)

            dout4 = self.dlayer4(dout5_out4)
            dout4_out3 = torch.cat([dout4, out3], 1)

            dout3 = self.dlayer3(dout4_out3)
            dout3_out2 = torch.cat([dout3, out2], 1)

            dout2 = self.dlayer2(dout3_out2)
            dout2_out1 = torch.cat([dout2, out1], 1)

            dout1 = self.dlayer1(dout2_out1)

            unet_embed_seq.append(dout1)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        unet_embed_seq = torch.stack(unet_embed_seq, dim=0).transpose_(0, 1)  # torch.Size([20, 10, 1, 32, 64])
        # unet_embed_seq: shape=(batch, time_step, input_size)

        layer_output, last_states = self.convlstm(unet_embed_seq)
        # layer_output[0]: (batch_size, time_step, output_channels, height, width) of the last layer
        last_time_of_last_layer_output = layer_output[-1].clone()
        return last_time_of_last_layer_output


if __name__ == "__main__":
    input_tensor = torch.randn(20, 10, 12, 32, 64)  # (batch_size, time_step, channels, height, width)
    model = UNet_ConvLSTM()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # torch.Size([20, 1, 32, 64])
