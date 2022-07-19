import torch.nn as nn
import MODEL.Block as BLK
import MODEL.Layer as LYR
import torch


class PAN_LIGHT2CAT(nn.Module) :
    def __init__(self, act_fn):
        super(PAN_LIGHT2CAT, self).__init__()

        self.cvt1 = LYR.ConvGN(256,256,1,1,0,act_fn,1)
        self.cvt2 = LYR.ConvGN(512, 256, 1, 1, 0, act_fn, 1)
        self.cvt3 = LYR.ConvGN(1024, 256, 1, 1, 0, act_fn, 1)

        self.up3t2= nn.Sequential(
            LYR.ConvGN(256,128,3,1,1,act_fn),
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)
        )
        self.up2t1 = nn.Sequential(
            LYR.ConvGN(384,256,3,1,1,act_fn),
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False)
        )

        self.down1t2 = LYR.ConvGN(512,128,3,2,1, act_fn,1)
        self.down2t3 = LYR.ConvGN(384, 256, 3, 2, 1, act_fn, 1)


    def forward(self, FM_1, FM_2, FM_3):
        NFM3 = self.cvt3(FM_3)
        NFM2 = torch.cat([self.cvt2(FM_2) , self.up3t2(NFM3)],dim=1)
        NFM1 = torch.cat([self.cvt1(FM_1) , self.up2t1(NFM2)],dim=1)

        DNFM1 = NFM1
        DNFM2 = torch.cat([NFM2 , self.down1t2(DNFM1)],dim=1)
        DNFM3 = torch.cat([NFM3 , self.down2t3(NFM2)],dim=1)

        return DNFM1, DNFM2, DNFM3

