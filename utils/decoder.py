import torch
import torch.nn as nn
from utils.resnet_block import ResnetBlock
from utils.attention import flash_SpatialSelfAttention,LinearSelfAttention
from utils.utils import Upsample,Normalize,nonlinearity
import numpy as np
from einops import rearrange
class Decoder(nn.Module):
    def __init__(self,*,ch,out_ch,ch_mult=(1,2,4,8),num_res_blocks,\
                attn_resolutions,dropout=0.0,resamp_with_conv=True,in_channels,\
                resolution,z_channels,give_pre_end=False,tanh_out=False,**kwargs
                ) -> None:
        super().__init__()
        self.ch=ch
        self.temb_ch=0
        self.num_resolutions=len(ch_mult)
        self.resolution=resolution
        self.in_channels=in_channels,
        self.give_pre_end=give_pre_end
        self.tanh_out=tanh_out
        self.num_res_block=num_res_blocks

        in_ch_mult=(1,)+tuple(ch_mult)
        block_in=ch*ch_mult[self.num_resolutions-1]
        curr_res=resolution//2**(self.num_resolutions)
        self.z_shape=(1,z_channels,curr_res,curr_res)
        #若指定了axis，把数组a沿着维度axis进行切片，并把切片后得到的一系列数组每个对应位置的元素相乘(element-wise product)，
        # 如果axis未指定，则对数组进行扁平化后计算所有元素的乘积.
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape))) 
        
        self.conv_in=torch.nn.Conv2d(z_channels,block_in,3,1,1)

        self.mid=nn.Module()
        self.mid.block_1=ResnetBlock(in_channels=block_in,out_channels=block_in,dropout=dropout,temb_channels=self.temb_ch)
        self.mid.attn_1=flash_SpatialSelfAttention(block_in)
        self.mid.block_2=ResnetBlock(in_channels=block_in,out_channels=block_in,dropout=dropout,temb_channels=self.temb_ch)

        self.up=nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block=nn.ModuleList()
            attn=nn.ModuleList()
            block_out=ch*ch_mult[i_level]
            for i_block in range(self.num_res_block+1):
                block.append(ResnetBlock(in_channels=block_in,out_channels=block_out,dropout=dropout,temb_channels=self.temb_ch))
                block_in=block_out

                if curr_res in attn_resolutions:
                    attn.append(flash_SpatialSelfAttention(block_in))
            up=nn.Module()
            up.block=block
            up.attn=attn
            if i_level !=-1:
                up.upsample=Upsample(block_in,resamp_with_conv)
                curr_res=curr_res*2
            self.up.insert(0,up)

        self.norm_out=Normalize(block_in)
        self.conv_out=torch.nn.Conv2d(block_in,out_ch,3,1,1)
    
    def forward(self,z):
        self.last_z_shape=z.shape

        temb=None

        h=self.conv_in(z)

        h=self.mid.block_1(h,temb)
        height=h.shape[2]
        # h=rearrange(h,'b c h w -> b (h w) c').contiguous()
        h=self.mid.attn_1(h)
        # h=rearrange(h,'b (h w) c -> b c h w ',h=height).contiguous()
        h=self.mid.block_2(h,temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_block+1):
                h=self.up[i_level].block[i_block](h,temb)
                if len(self.up[i_level].attn)>0:
                    # height=h.shape[2]
                    # h=rearrange(h,'b c h w -> b (h w) c').contiguous()
                    h=self.up[i_level].attn[i_block](h)
                    # h=rearrange(h,'b (h w) c -> b c h w ',h=height).contiguous()
            if i_level !=-1:
                h=self.up[i_level].upsample(h)
        
        if self.give_pre_end:
            return h
        
        h=self.norm_out(h)
        h=nonlinearity(h)
        h=self.conv_out(h)
        if self.tanh_out:
            h=torch.tanh(h)
        return h