import torch.nn as nn


import MODEL.Module as MDU



class CBAM(nn.Module) :
    def __init__(self, in_dim, compress=16):
        super(CBAM, self).__init__()
        self.ca = MDU.Module_CA(in_dim, compress)
        self.sa = MDU.Module_SA()


    def forward(self,x):
        #print(self.ca(x).shape)


        out = x * self.ca(x)
        #print(self.sa(out).shape)
        out = out * self.sa(out)

        return out

