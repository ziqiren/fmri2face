import time

import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm1d, PReLU, Sequential
from einops import rearrange

class fmri2map(nn.Module):
    def __init__(self, vec_dim):
        super(fmri2map, self).__init__()
        self.input_layer_1 = Sequential(Linear(vec_dim, 64),
                                        BatchNorm1d(64),
                                        PReLU(64),
                                        Linear(64, 256 * 7 * 7),
                                        BatchNorm1d(256 * 7 * 7),
                                        PReLU(256 * 7 * 7),
                                        )
        self.input_layer_2 = Sequential(Linear(vec_dim, 64),
                                        BatchNorm1d(64),
                                        PReLU(64),
                                        Linear(64, 128 * 7 * 7),
                                        BatchNorm1d(128 * 7 * 7),
                                        PReLU(128 * 7 * 7),
                                        )
        self.input_layer_3 = Sequential(Linear(vec_dim, 64),
                                        BatchNorm1d(64),
                                        PReLU(64),
                                        Linear(64, 64 * 7 * 7),
                                        BatchNorm1d(64 * 7 * 7),
                                        PReLU(64 * 7 * 7),
                                        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
    def forward(self, input):
        out1 = self.input_layer_1(input)
        out2 = self.input_layer_2(input)
        out3 = self.input_layer_3(input)
        return out1,out2,out3

class map2style(nn.Module):
    def __init__(self):
        super(map2style, self).__init__()
        self.output_layer_3 = nn.Linear(256 * 7 * 7, 512 * 9)
        self.output_layer_4 = nn.Linear(128 * 7 * 7, 512 * 5)
        self.output_layer_5 = nn.Linear(64 * 7 * 7, 512 * 4)
    def forward(self, input):
        lc_part_2 = self.output_layer_3(input[0]).view(-1, 9, 512)
        lc_part_3 = self.output_layer_4(input[1]).view(-1, 5, 512)
        lc_part_4 = self.output_layer_5(input[2]).view(-1, 4, 512)
        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x, lc_part_2, lc_part_3, lc_part_4

class adapter(nn.Module):
    def __init__(self):
        super(adapter, self).__init__()
        self.output_layer_3 = nn.Linear(256 * 7 * 7, 512 * 9)
    def forward(self, x, input, type='high',lam=0.9):
        if type == 'high':
            lc_part_2 = self.output_layer_3(input[0]).view(-1, 9, 512)
            return lam * x + (1. - lam) * lc_part_2