import torch.nn as nn
import MODEL.Block as BLK

class Head_YVX_SEG11(nn.Module):
    def __init__(self, dim1, dim2, dim3, act_fn):
        super(Head_YVX_SEG11, self).__init__()

        # Only One
        self.head1 = BLK.Head_YOLOX(dim1, act_fn)
        self.head2 = BLK.Head_YOLOX(dim2, act_fn)
        self.head3 = BLK.Head_YOLOX(dim3, act_fn)

        self.seg_head1 = BLK.One_BOTTLE1_Block(dim1, 128, 0,act_fn)
        self.seg_head2 = BLK.One_BOTTLE1_Block(dim2, 128, 1, act_fn)
        self.seg_head3 = BLK.One_BOTTLE1_Block(dim3, 128, 2, act_fn)



    def forward(self, DNFM1, DNFM2, DNFM3):
        PRD1 = self.head1(DNFM1)
        PRD2 = self.head2(DNFM2)
        PRD3 = self.head3(DNFM3)

        SEG_FM1 = self.seg_head1(DNFM1)
        SEG_FM2 = self.seg_head2(DNFM2)
        SEG_FM3 = self.seg_head3(DNFM3)

        return PRD1, PRD2, PRD3, SEG_FM1, SEG_FM2, SEG_FM3


