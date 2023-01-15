import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet



class ProtoNet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        
        #self.fc = torch.nn.Linear(640**2, 640)

        self.conv = nn.Conv2d(640, 25, kernel_size=3, stride=1,
                     padding=1, bias=False)

        self.relu = nn.LeakyReLU(0.1)
    def forward(self, input):
        if self.mode == 'encoder':
            return self.encode(input)
        elif self.mode == 'distance':
            spt, qry = input
            return self.distance(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def distance(self, spt, qry):
      
      #  spt = spt.squeeze(0)

        #qry = qry.mean(dim=[-1, -2])
        #spt = spt.mean(dim=[-1, -2])

        spt    = spt.mean(2) 
        
        qry = qry.transpose(1,2).reshape(qry.size(0),self.args.way*self.args.query, -1)
        qry = F.normalize(qry, dim=-1)
        spt = F.normalize(spt, dim=-1)
        similarity_matrix = torch.bmm(qry, spt.transpose(1,2))


        
        return similarity_matrix / self.args.temperature
    


    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x
