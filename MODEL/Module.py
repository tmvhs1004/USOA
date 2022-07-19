import torch
import torch.nn as nn
import MODEL.Layer as LYR

class Module_DK(nn.Module) :
    def __init__(self,in_dim, out_dim,k_size, pad ,act_fn, dilation=1  ):
        super(Module_DK, self).__init__()

        self.main = nn.Sequential(
            LYR.Conv_Basic(in_dim, out_dim, 1, 1, 0 , act_fn, dilation=dilation),
            LYR.Conv_Basic(out_dim, in_dim, k_size, 1, pad, act_fn, dilation=dilation)

        )

    def forward(self, x):
        return self.main(x) + x





class Module_CA(nn.Module) :
    def __init__(self, in_dim, compress=16):
        super(Module_CA, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, in_dim //compress, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Mish(),
            nn.Conv2d(in_dim // compress, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        )


        self.sig = nn.Sigmoid()

    def forward(self, x):
        p1 = self.avg(x)
        p2 = self.max(x)

        p1 = self.main(p1)
        p2 = self.main(p2)

        out = p1 + p2
        out = self.sig(out)

        return out

class Module_SA(nn.Module) :
    def __init__(self):
        super(Module_SA, self).__init__()

        self.conv1= nn.Conv2d(2, 1, 7, padding=3, bias= False)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        p1 = torch.mean(x, dim=1, keepdim=True)
        p2, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([p1, p2], dim=1)
        out = self.conv1(out)
        out = self.sig(out)

        return out



