import torch
from torch import nn
import math

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

    def linear(self):
        P_mat = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
        _, sP, _ = torch.svd(P_mat)
        sv = sP[0]

        return self.conv.weight, sv

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


        return conv21.permute([1, 0, 2, 3])

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





class LIH_Subnet(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):     
        super(LIH_Subnet, self).__init__()

        self.dilate = dilate
        self.predictor = WLBlock(pin_ch, uin_ch,f_ch,f_sz,num_layers,dilate)
        self.updator = WLBlock(uin_ch, pin_ch,f_ch,f_sz,num_layers,dilate)

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




class LIH(nn.Module):
    def __init__(self, pin_ch, uin_ch, num_scale, f_ch, f_sz, dilate, num_layers):  
        super(LIH, self).__init__()
        self.layers = []
        for _ in range(num_scale):
            self.layers.append(LIH_Subnet(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
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



# Lifting based Invertible Hiding (LIH) Network 
class Model(nn.Module):
    def __init__(self, pin_ch, uin_ch, num_step, pu_ch =32, kernel_size = 3, dilate = 1,num_layers = 4 ):
        super(Model, self).__init__()
        
        self.model = LIH(pin_ch, uin_ch, num_step, pu_ch, kernel_size, dilate, num_layers)

    def forward(self, xc, xd, rev=False):

        if not rev:
            outc, outd = self.model.forward(xc, xd)

        else:
            outc, outd = self.model.inverse(xc, xd)

        return outc, outd