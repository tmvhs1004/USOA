import MODEL.Block as BLK
import torch.nn as nn
import MODEL.SPP as SPP



class BodyDevBU6NA(nn.Module) :
    def __init__(self, act_fn):
        super(BodyDevBU6NA, self).__init__()
        self.block_in = BLK.DK53_Inx3(act_fn=act_fn,k_size=3, pad=1, dilation=1 )
        self.block_1 = BLK.DK53ATBlock(32,  64,  3, 1, 1, act_fn, dilation=1)
        self.block_2 = BLK.DK53ATBlock(64,  128, 3, 1, 2, act_fn, dilation=1)
        self.block_3 = BLK.DK53ATBlock(128, 256, 3, 1, 8, act_fn, dilation=1)
        self.block_4 = BLK.DK53ATBlock(256, 512, 3, 1, 8, act_fn, dilation=1)
        self.block_5 = BLK.DK53ATBlock(512, 1024,3, 1, 4, act_fn, dilation=1)
        self.spp = SPP.ASPP_Y5(1024, act_fn)

    def forward(self,x):
        FM_1 = self.block_3(self.block_2(self.block_1(self.block_in(x))))
        FM_2 = self.block_4(FM_1)
        FM_3 = self.block_5(FM_2)
        FM_3 = self.spp(FM_3)
        return FM_1, FM_2, FM_3

