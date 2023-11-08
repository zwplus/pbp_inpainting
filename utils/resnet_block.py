import torch.nn as nn
import torch
from utils.utils import Normalize,nonlinearity




class ResnetBlock(nn.Module):
    def __init__(self,*,in_channels,out_channels=None,conv_shortcut=False,\
                dropout=0.,temb_channels=0) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels if out_channels is not None else in_channels
        self.use_conv_shortcut=conv_shortcut

        self.norm1=Normalize(in_channels)
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        if temb_channels >0:
            self.temb_proj=torch.nn.Linear(temb_channels,out_channels)

        self.norm2=Normalize(out_channels)
        self.dropout=torch.nn.Dropout(dropout)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

        if self.in_channels != self.out_channels :
            if self.use_conv_shortcut:
                self.conv_shortcut=nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride=1,padding=1)
            else:
                self.nin_shortcut=nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x,temb=None):
        h=x
        h=self.norm1(h)
        h=nonlinearity(h)
        h=self.conv1(h)

        if temb is not None:
            h=h+self.temb_proj(temb)[:,:,None,None]
        
        h=self.norm2(h)
        h=nonlinearity(h)
        h=self.dropout(h)
        h=self.conv2(h)

        if self.in_channels!=self.out_channels:
            if self.use_conv_shortcut:
                x=self.conv_shortcut(x)
            else:
                x=self.nin_shortcut(x)
        
        return x+h




            
