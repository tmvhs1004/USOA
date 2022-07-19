import torch
import torch.nn as nn
import MODEL.Layer as LYR
import MODEL.Module as MDU
import Config as cfg
import MODEL.CBAM as CB






class DK53_Inx3(nn.Module) :

    def __init__(self,act_fn, k_size =3, pad=1,dilation=1, out_dim=32):
        super(DK53_Inx3, self).__init__()
        self.main = nn.Sequential(
            LYR.Conv_Basic(3, out_dim, k_size, 1, pad, act_fn, dilation=dilation),
            LYR.Conv_Basic(out_dim, out_dim, k_size, 1, pad, act_fn, dilation=dilation),
            LYR.Conv_Basic(out_dim, out_dim, k_size, 1, pad, act_fn, dilation=dilation)
        )


    def forward(self , x):
        return self.main(x)





class DK53ATBlock(nn.Module) :
    def __init__(self, in_dim, out_dim ,k_size,pad, MDU_num ,act_fn  ,dilation, stride=2):
        super(DK53ATBlock, self).__init__()

        self.main = nn.Sequential(
            LYR.Conv_Basic(in_dim, out_dim , k_size, stride, pad, act_fn,dilation=dilation)
        )

        for idx in range(0, MDU_num) :
            name = 'DART_' + str(idx)
            self.main.add_module(name, MDU.Module_DK(out_dim, in_dim,k_size, pad, act_fn, dilation))

        self.main.add_module('last_ATT',CB.CBAM(out_dim))

    def forward(self, x):
        return self.main(x)



class Head_YOLOX(nn.Module) :
    def __init__(self,in_dim, act_fn):
        super(Head_YOLOX, self).__init__()

        self.conv = LYR.ConvGN(in_dim, 256, 1,1,0, act_fn)

        self.p1 = nn.Sequential(
            LYR.ConvGN(256, 256, 3, 1, 1, act_fn),
            LYR.ConvGN(256, 256, 3, 1, 1, act_fn)
        )

        self.p2 = nn.Sequential(
            LYR.ConvGN(256, 256, 3, 1, 1, act_fn),
            LYR.ConvGN(256, 256, 3, 1, 1, act_fn)
        )

        self.cls_head = nn.Sequential(
            CB.CBAM(256),
            nn.Conv2d(256, cfg.NUM_CLASSES * 3, 1, 1, 0, bias=False)
        )

        self.obj_head =nn.Sequential(
            CB.CBAM(256),
            nn.Conv2d(256, 3, 1, 1, 0, bias=False)
        )

        self.box_head = nn.Sequential(
            CB.CBAM(256),
            nn.Conv2d(256, 12, 1, 1, 0, bias=False)
        )


    def forward(self, x):


        out = self.conv(x)
        p1 = self.p1(out)
        p2 = self.p2(out)


        cls = self.cls_head(p1)\
            .reshape(cfg.BATCH_SIZE, 3, cfg.NUM_CLASSES , x.shape[2], x.shape[3])\
            .permute(0, 1, 3, 4, 2)


        box = self.box_head(p2) \
            .reshape(cfg.BATCH_SIZE, 3, 4, x.shape[2], x.shape[3]) \
            .permute(0, 1, 3, 4, 2)

        obj = self.obj_head(p2) \
            .reshape(cfg.BATCH_SIZE, 3, 1, x.shape[2], x.shape[3]) \
            .permute(0, 1, 3, 4, 2)





        return torch.cat([box, obj, cls], dim=4)




class One_BOTTLE1_Block(nn.Module) :
    def __init__(self,in_dim, out_dim, r_num, act_fn):
        super(One_BOTTLE1_Block, self).__init__()
        # print(in_dim)
        self.main = nn.Sequential(
            LYR.ConvGN(in_dim, out_dim//2, 1, 1, 0, act_fn),
            LYR.ConvGN(out_dim//2 , out_dim , 3, 1, 1, act_fn),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        for idx in range(0, r_num) :
            name = 'BOT1_' + str(idx)
            self.main.add_module(name, LYR.ConvGN(out_dim, out_dim//2, 1,1,0, act_fn))

            name = 'BOT2_' + str(idx)
            self.main.add_module(name, LYR.ConvGN(out_dim//2, out_dim, 3, 1, 1, act_fn))

            name = 'MPBU_' + str(idx)
            self.main.add_module(name, nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=False))

        self.main.add_module('last2', nn.Conv2d(out_dim, cfg.SEG_CLASSES, 1, 1, 0))
        self.main.add_module('last3', nn.Upsample(scale_factor=4, mode = 'bilinear', align_corners=False))


    def forward(self, x):
        # print('BOT')
        return self.main(x)





