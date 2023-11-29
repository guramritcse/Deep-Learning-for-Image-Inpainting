import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn

"""
Generator Network
"""
class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.coarse_generator = CoarseGenerator(config['input_dim'], config['ngf'], use_cuda, device_ids)
        self.fine_generator = CoarseGenerator(config['input_dim'], config['ngf'], use_cuda, device_ids)

    def forward(self, img, mask, mask1):
        # Mask 1 is essentially with no holes
        coarse_img = self.coarse_generator(img, mask)
        fine_img = self.fine_generator(img, mask1)
        return coarse_img, fine_img

"""
Coarse Generator Network
"""
class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)
        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, img, mask):
        ones = torch.ones(img.size(0), 1, img.size(2), img.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        img = self.conv1(torch.cat([img, ones, mask], dim=1))
        img = self.conv2_downsample(img)
        img = self.conv3(img)
        img = self.conv4_downsample(img)
        img = self.conv5(img)
        img = self.conv6(img)
        img = self.conv7_atrous(img)
        img = self.conv8_atrous(img)
        img = self.conv9_atrous(img)
        img = self.conv10_atrous(img)
        img = self.conv11(img)
        img = self.conv12(img)
        img = F.interpolate(img, scale_factor=2, mode='nearest')
        img = self.conv13(img)
        img = self.conv14(img)
        img = F.interpolate(img, scale_factor=2, mode='nearest')
        img = self.conv15(img)
        img = self.conv16(img)
        img = self.conv17(img)
        img_coarse = torch.clamp(img, -1., 1.)

        return img_coarse

"""
Local Discriminator Network
"""
class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, img):
        img = self.dis_conv_module(img)
        img = img.view(img.size()[0], -1)
        img = self.linear(img)

        return img

"""
Global Discriminator Network
"""
class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, img):
        img = self.dis_conv_module(img)
        img = img.view(img.size()[0], -1)
        img = self.linear(img)

        return img

"""
Discriminator Convolutional Module
"""
class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, img):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = self.conv4(img)

        return img


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)

"""
Convolutional Block
"""
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None

        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        else:
            self.norm = None

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = None

        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, img):
        if self.pad:
            img = self.conv(self.pad(img))
        else:
            img = self.conv(img)
        if self.norm:
            img = self.norm(img)
        if self.activation:
            img = self.activation(img)
        return img
