import torch.nn as nn


class Conv_Basic(nn.Module) :

    def __init__(self, in_dim, out_dim, k_size, stride, pad, act_fn , group=1,dilation=1 ):
        super(Conv_Basic, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=k_size, stride=stride, padding=pad, groups=group, dilation=(dilation,dilation), bias=False),
            #nn.BatchNorm2d(out_dim),
            nn.GroupNorm(32, out_dim),
            act_fn
            #nn.Dropout2d(p=0.2)
        )


    def forward(self, x):

        return self.main(x)

class ConvGN(nn.Module) :
    def __init__(self,in_dim, out_dim, k_size, stride, pad, act_fn, group=1 ,dilation=1):
        super(ConvGN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=k_size, stride=stride, padding=pad, groups=group, dilation=(dilation,dilation), bias=False),
            nn.GroupNorm(32,out_dim),
            act_fn
            #nn.Dropout2d(p=0.2)
        )


    def forward(self, x):
        # print(x.shape)
        return self.main(x)
