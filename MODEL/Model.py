import torch.nn as nn
import torch

import MODEL.Body as BDY
import MODEL.Neck as NCK
import MODEL.Head as HAD



class USOA(nn.Module):
    def __init__(self):
        super(USOA, self).__init__()
        self.body = BDY.BodyDevBU6NA(nn.Mish())
        self.neck = NCK.PAN_LIGHT2CAT(nn.Mish())
        self.head = HAD.Head_YVX_SEG11(512, 512, 512, nn.Mish())

    def forward(self, x):
        FM_1, FM_2, FM_3 = self.body(x)
        HFM_1, HFM_2, HFM_3 = self.neck(FM_1, FM_2, FM_3)
        PRD1, PRD2, PRD3, SEG_FM1, SEG_FM2, SEG_FM3 = self.head(HFM_1, HFM_2, HFM_3)


        return PRD1, PRD2, PRD3, SEG_FM1, SEG_FM2, SEG_FM3

