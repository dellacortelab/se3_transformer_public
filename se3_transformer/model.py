import numpy as np

import torch
from torch import nn

from se3_transformer.se3_transformer import MultiHead, GNonlinear


class Model(nn.Module):
    def __init__(
            self,
            heads=8,
            c_in=4,
            c=8,
            c_hid=2,
            c_out=1,
            depth=3,
            d_in=0,
            d_hid=2,
            d_out=1,
            rdc=torch.sum
    ):
        super(Model, self).__init__()
        layers = []
        
        layers.append(MultiHead(d_in=d_in, d_out=d_hid, c_in=c_in, c_hid=c_hid, c_out=c, heads=heads, rdc=rdc))
        layers.append(GNonlinear(d_in=d_hid, c_in=c))

        for _ in range(depth-2):
            layers.append(MultiHead(d_in=d_hid, d_out=d_hid, c_in=c, c_hid=c_hid, c_out=c, heads=heads, rdc=rdc))
            layers.append(GNonlinear(d_in=d_hid, c_in=c))

        layers.append(MultiHead(d_in=d_hid, d_out=d_out, c_in=c, c_hid=c_hid, c_out=c_out, heads=heads, rdc=rdc))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, pc, f):
        out = f
        for i, layer in enumerate(self.layers):
            out = layer(pc, out)
        return out
