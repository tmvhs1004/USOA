import torch
import torch.nn as nn
import MODEL.Layer as LYR


class ASPP_Y5(nn.Module) :
    def __init__(self, in_dim, act_fn):
        super(ASPP_Y5, self).__init__()
        self.main = nn.Sequential(
            LYR.ConvGN(in_dim, in_dim // 2, 1, 1, 0, act_fn),
            ASPP_UNIT(in_dim//2, act_fn),
            LYR.ConvGN(in_dim * 2, in_dim, 1, 1, 0, act_fn)
        )

    def forward(self, x):
        return self.main(x)



class ASPP_UNIT(nn.Module) :
    def __init__(self,dim,  act_fn):
        super(ASPP_UNIT, self).__init__()
        self.cp1 = LYR.ConvGN(dim, dim, 3, 1, 6, act_fn, 1, 6)
        self.cp2 = LYR.ConvGN(dim, dim, 3, 1, 12, act_fn, 1, 12)
        self.cp3 = LYR.ConvGN(dim, dim, 3, 1, 18, act_fn, 1, 18)
        self.cp4 = LYR.ConvGN(dim, dim, 3, 1, 24, act_fn, 1, 24)

    def forward(self, x):

        p1 = self.cp1(x)
        p2 = self.cp2(x)
        p3 = self.cp3(x)
        p4 = self.cp4(x)

        return torch.cat([p1,p2,p3,p4], dim=1)