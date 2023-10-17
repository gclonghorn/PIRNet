import torch
from torch import nn
import math
from util import waveletDecomp
from SRM import SRM

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, f_sz, dilate):
        super(Conv2d, self).__init__()
        if_bias = False

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias)
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        return x


class SepConv(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(SepConv, self).__init__()

        if_bias = False
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,     #depth-with conv
                               groups=in_ch)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0.0, mode='fan_out', nonlinearity='relu')

        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=1,
                               padding=math.floor(1 / 2) + dilate - 1, dilation=dilate, bias=if_bias,        #1*1 conv
                               groups=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0.0, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ResBlockSepConv(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResBlockSepConv, self).__init__()

        self.conv1 = SepConv(in_ch, f_ch, f_sz, dilate)
        self.conv2 = SepConv(f_ch, in_ch, f_sz, dilate)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.identity = torch.zeros([in_ch, in_ch, 2 * f_sz - 1, 2 * f_sz - 1], device=torch.device('cuda:0'))
        for i in range(in_ch):
            self.identity[i, i, int(f_sz) - 1, int(f_sz) - 1] = 1

    def forward(self, x):
        return self.leaky_relu(x + self.conv2(self.leaky_relu(self.conv1(x))))


class WLBlock(nn.Module):
    def __init__(self, in_ch, out_ch, f_ch, f_sz, num_layers, dilate):
        super(WLBlock, self).__init__()

        if_bias = False
        self.layers = []
        self.layers.append(Conv2d(in_ch, f_ch, f_sz, dilate))
        for _ in range(int(num_layers)):
            self.layers.append(ResBlockSepConv(f_ch, int(f_ch / 1), 2 * f_sz - 1, dilate))
        self.net = mySequential(*self.layers)

        self.convOut = nn.Conv2d(f_ch, out_ch, f_sz, stride=1, padding=math.floor(f_sz / 2) + dilate - 1,
                                 dilation=dilate, bias=if_bias)
        self.convOut.weight.data.fill_(0.)

    def forward(self, x):
        x = self.net(x)
        out = self.convOut(x)
        return out



class LSR_Subnet(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):     
        super(LSR_Subnet, self).__init__()

        self.dilate = dilate

        pf_ch = int(f_ch)
        uf_ch = int(f_ch)
        self.predictor = WLBlock(pin_ch, uin_ch, pf_ch, f_sz, num_layers, dilate)
        self.updator = WLBlock(uin_ch, pin_ch, uf_ch, f_sz, num_layers, dilate)

    def forward(self, xc, xd):
        Fxc = self.predictor(xc)
        xd = - Fxc + xd
        Fxd = self.updator(xd)
        xc = xc + Fxd

        return xc, xd

    def inverse(self, xc, xd):
        Fxd = self.updator(xd)
        xc = xc - Fxd
        Fxc = self.predictor(xc)
        xd = xd + Fxc

        return xc, xd



class LSR(nn.Module):
    def __init__(self, pin_ch, uin_ch, num_step, f_ch, f_sz, dilate, num_layers):  
        super(LSR, self).__init__()
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LSR_Subnet(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
        self.net = mySequential(*self.layers)

        dilate = 1

    def forward(self, xc, xd):
        for i in range(len(self.net)):
            outc, outd = self.net[i].forward(xc, xd)
        return outc, outd

    def inverse(self, xc, xd):
        for i in reversed(range(len(self.net))):
            outc, outd = self.net[i].inverse(xc, xd)
        return outc, outd



# Lifting-based Secure Restoration (LSR)
class Model(nn.Module):
    def __init__(self, steps=4, layers=4,  klvl=3, dnlayers=3,mid=12,enc=[2,2,4,8],dec=[2,2,2,2]):
        super(Model, self).__init__()
        pin_chs = 3
        uint_chs = 9
        nstep = steps
        nlayer = layers

        self.innlayers = []
        for ii in range(klvl):
            dilate = 2 ** ii
            if ii > 1:
                dilate = 2
            self.innlayers.append(LSR(pin_ch=pin_chs, uin_ch=uint_chs,num_step=nstep,f_ch=32,f_sz=3, dilate=dilate,num_layers=nlayer))
        self.innnet = mySequential(*self.innlayers)
        #Secure restoration Module(SRM)
        self.ddnlayers = []
        for ii in range(klvl):
            self.ddnlayers.append(SRM(img_channel=9,width=32,middle_blk_num=mid,enc_blk_nums=enc,dec_blk_nums=dec))

        self.ddnnet = mySequential(*self.ddnlayers)
        self.split = waveletDecomp()

    def forward(self,x,sp1=4,sp2=2,sp3=4):
        xc, xd, xc_, xd_  = [], [], [], []

        for i in range(len(self.innnet)):
           
            if i == 0:
                xcc , xdd = self.split.forward(x)
                tmpxc, tmpxd = self.innnet[i].forward(xcc,xdd)
            else:
                xcc, xdd = self.split.forward(xc[i - 1])
                tmpxc, tmpxd = self.innnet[i].forward(xcc,xdd)

            xc.append(tmpxc)
            xd.append(tmpxd)
            tmpxd_ = self.ddnnet[i].forward(xd[i],sp1=sp1,sp2=sp2,sp3=sp3)

            xd_.append(tmpxd_)
            xc_.append(tmpxc)
        for i in reversed(range(len(self.innnet))):
            if i > 0:
                tmpxc,tmpxd = self.innnet[i].inverse(xc_[i], xd_[i])
                xc_[i - 1] = self.split.inverse(tmpxc,tmpxd)
            else:
                tmpxc,tmpxd = self.innnet[i].inverse(xc_[i], xd_[i])
                out = self.split.inverse(tmpxc,tmpxd)

        return out
