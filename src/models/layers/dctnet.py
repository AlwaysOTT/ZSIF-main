from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.dct_norm = nn.LayerNorm([96], eps=1e-6)
        self.dropout = nn.Dropout(0.2)
        # self.dct_norm = nn.LayerNorm([96], eps=1e-6)#for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        # print("stack_dct:", stack_dct.shape)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''

        # lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        # print("lr_weight:", lr_weight.shape)
        lr_weight = self.dct_norm(lr_weight)
        # print("lr_weight:", lr_weight.shape)
        x = x + self.dropout(x * lr_weight)

        # print("lr_weight",lr_weight.shape)
        return self.dct_norm(x)  # result


# class senet_block(nn.Module):
#     def __init__(self, channel=512, ratio=1):
#         super(dct_channel_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // 4, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel //4, channel, bias=False),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         # b, c, l = x.size() # (B,C,L)
#         # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)
#         # print("y",y.shape)
#         x = x.permute(0,2,1)
#         b, c, l = x.size()
#         y = self.avg_pool(x).view(b, c) # (B,C,L) ->(B,C,1)
#         # print("y",y.shape)
#         # y = self.fc(y).view(b, c, 96)

#         y = self.fc(y).view(b,c,1)
#         # print("y",y.shape)
#         # return x * y
#         return (x*y).permute(0,2,1)

if __name__ == '__main__':
    device = torch.device('npu:{}'.format(1))
    tensor = torch.rand(8, 7, 96)
    dct_model = dct_channel_block(96)
    result = dct_model.forward(tensor)
    print("result.shape:", result.shape)
