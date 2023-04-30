import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
import statistics
import math

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class CtrlbDropout2D(nn.Module):
    def __init__(self,p=0.1,active = True):
        super(CtrlbDropout2D, self).__init__()       
        self._p = p
        self._active = active
        self.drops = 0

    def _tensor_to_output(self,tensor):

      return torch.bernoulli(1-tensor)
    
    
    def _assembleCtrlb(self,x):
      # x: [B, C, H, W]
      x_pooled = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1])).squeeze() # x_pooled: [B, C]
      g = x_pooled**2
      orderedS = torch.abs(g)**0.5

    #   max_vals, idx = torch.topk(orderedS, math.ceil(self._p*orderedS.shape[1]),dim = 1, sorted=False)
    #   m = torch.mean(max_vals,dim=1).unsqueeze(1)
    #   prob = torch.clamp(orderedS/(m),0,1)

      prob = (orderedS/(torch.max(orderedS,dim=1)[0].unsqueeze(1)))-0.05
      prob = torch.clamp(prob,0,1)

      
      
      # g = torch.diag_embed(x).to(device)      
      # r = torch.svd(g,compute_uv=True)
      # us =  torch.bmm(r.U,torch.diag_embed(r.S))
      # norm = torch.norm(us,dim=1).unsqueeze(1)
      # orderedS = torch.bmm(norm,r.V.mT).squeeze()

      # prob = torch.softmax(r.S,1)

      # su  = torch.diagonal((torch.diag_embed(r.S)*r.U),dim1=1, dim2=2)
      # su = r.U*r.S
      # prob = torch.softmax(orderedS,1)
      # prob = torch.exp(-orderedS/torch.max(orderedS))
    #   prob = torch.softmax(-orderedS/torch.max(orderedS),1)




      #  u, s, v  = svd(x)
      #  prob = torch.softmax(s,1)
      return self._tensor_to_output(prob)

    def forward(self,x):
      H, W = x.shape[-2:]
      if not self.training:
          return x
      
      with torch.no_grad():
         drop = self._assembleCtrlb(x)
         drop = drop.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
         self.drops = (self.drops + (torch.mean((torch.sum(drop, dim=1)/drop.shape[1])))) / 2.0
        #  print (self.drops,"\t",  torch.mean((torch.sum(drop, dim=1)/drop.shape[1])))
         
        #  self.drops /= self._iter
      return x*drop
    

a = torch.randn(8, 16, 7, 7)
dp2d = CtrlbDropout2D()

dp2d(a)