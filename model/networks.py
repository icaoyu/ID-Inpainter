import numpy as np
import torch
import torch.nn as nn

class AttributesEncoder(nn.Module):
    def __init__(self):
        super(AttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512, 512]
        self.Encoder = nn.ModuleDict()

        self.Encoder.add_module(name=f'layer_0',module=nn.Sequential(
            nn.UpsamplingBilinear2d(size=(128,128)),
            nn.Conv2d(self.Encoder_channel[0], self.Encoder_channel[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Encoder_channel[1]),
            nn.LeakyReLU(0.1)
        ))
        for i in range(1,6):
            self.Encoder.add_module(name=f'layer_{i}',module=nn.Sequential(
            nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i + 1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.Encoder_channel[i + 1]),
            nn.LeakyReLU(0.1)
        ))

        self.Decoder_inchannel = [512, 1024, 512, 256, 128]
        self.Decoder_outchannel = [512, 256, 128, 64, 32]

        self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            )for i in range(5)})

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Upsample112 = nn.UpsamplingBilinear2d(size=(112,112))

    def forward(self, x):
        arr_x = []
        for i in range(6):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)
        arr_y = []
        arr_y.append(arr_x[5])
        y = arr_x[5]
        for i in range(5):
            y = self.Decoder[f'layer_{i}'](y)
            y = torch.cat((y, arr_x[4-i]), 1)
            arr_y.append(y)
        y = self.Upsample112(y)
        arr_y.append(y)

        return arr_y


class AIF(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=256):
        super(AIF, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))

        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)

        i = r_id*h_bar + beta_id

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        a = r_att * h_bar + beta_att

        h_out = (1-m)*a + m*i

        return h_out


class AIFBlock(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, h_outchannel):
        super(AIFBlock, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel

        self.add1 = AIF(h_inchannel, z_inchannel)
        self.add2 = AIF(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        if not self.h_inchannel == self.h_outchannel:
            self.add3 = AIF(h_inchannel, z_inchannel)
            self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att, z_id):
        x1 = self.activation(self.add1(h_in, z_att, z_id))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att, z_id))
        x1 = self.conv2(x1)

        x2 = h_in
        if not self.h_inchannel == self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att, z_id))
            x2 = self.conv3(x2)

        return x1 + x2


class Generator(nn.Module):
    def __init__(self, z_id_size,normtype=True):
        super(Generator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 512, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Upsample112 = nn.UpsamplingBilinear2d(size=(112,112))
        self.h_inchannel = [512, 512, 512, 512, 256, 128, 64]
        self.z_inchannel = [512, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [512, 512, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : AIFBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(7)})
        if normtype:
            self.out = nn.Tanh()
        else:
            self.out = nn.Sigmoid()

    def forward(self, z_id, z_att):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
        for i in range(5):
            x = self.model[f"layer_{i}"](x, z_att[i], z_id)
            x = self.Upsample(x)
        x = self.model["layer_5"](x, z_att[5], z_id)
        x = self.Upsample112(x)
        x = self.model["layer_6"](x, z_att[6], z_id)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(Discriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


if __name__=='__main__':
    z_id = torch.rand(size=(1, 256))
    z_att = []
    z_att.append(torch.rand((1,512,2,2)))
    z_att.append(torch.rand((1,1024,4,4)))
    z_att.append(torch.rand((1,512,8,8)))
    z_att.append(torch.rand((1,256,16,16)))
    z_att.append(torch.rand((1,128,32,32)))
    z_att.append(torch.rand((1,64,64,64)))
    z_att.append(torch.rand((1,64,112,112)))
    x =torch.rand((1,3,112,112))
    net = Generator(256)
    attrnet = AttributesEncoder()
    netout = net(z_id,attrnet(x))
    print(netout.size())

