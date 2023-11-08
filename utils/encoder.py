import torch.nn as nn
from einops import rearrange
from utils.resnet_block import ResnetBlock
from utils.attention import flash_SpatialSelfAttention,LinearSelfAttention
from utils.utils import Downsample,Normalize,nonlinearity
class Encoder(nn.Module):
    def __init__(self,*,ch,out_ch,ch_mult=[1,2,4,8],num_res_blocks,attn_resolutions,dropout=0.0,\
                resamp_with_conv=True,in_channels,resolution,z_channels,double_z=True,
                ) -> None:
        super().__init__()

        self.ch=ch
        self.temb_ch=0
        self.num_resolutions=len(ch_mult)
        self.num_res_blocks=num_res_blocks
        self.resolution=resolution
        self.in_channles=in_channels

        self.conv_in=nn.Conv2d(self.in_channles,self.ch,3,1,1)
        curr_res=resolution
        in_ch_mult=(1,)+tuple(ch_mult)
        self.in_ch_mult=in_ch_mult
        self.down=nn.ModuleList()
        # print(in_ch_mult)

        for i_level in range(self.num_resolutions):
            block=nn.ModuleList()
            attn=nn.ModuleList()
            block_in=ch*in_ch_mult[i_level]
            block_out=ch*in_ch_mult[i_level+1]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,out_channels=block_out,temb_channels=self.temb_ch,dropout=dropout))

                block_in=block_out

                if curr_res in attn_resolutions:
                    attn.append(flash_SpatialSelfAttention(block_in))
            down=nn.Module()
            down.block=block
            down.attn=attn

            if i_level !=self.num_resolutions:
                down.downsample=Downsample(block_in,resamp_with_conv)
                curr_res=curr_res//2

            self.down.append(down)
        
        self.mid=nn.Module()
        self.mid.block_1=ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout=dropout)
        self.mid.attn_1=flash_SpatialSelfAttention(block_in)
        self.mid.block_2=ResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_ch,dropout=dropout)

        self.norm_out=Normalize(block_in)
        self.conv_out=nn.Conv2d(block_in,z_channels*2 if double_z else z_channels,kernel_size=3,stride=1,padding=1)
    
    def forward(self,x):
        temb=None

        hs=[self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h=self.down[i_level].block[i_block](hs[-1],temb)
                if len(self.down[i_level].attn)>0:
                    # height=h.shape[2]
                    # h=rearrange(h,'b c h w -> b (h w) c').contiguous()
                    h=self.down[i_level].attn[i_block](h)
                    # h=rearrange(h,'b (h w) c -> b c h w ',h=height).contiguous()
                hs.append(h)
            if i_level != self.num_resolutions:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h=hs[-1]
        h=self.mid.block_1(h,temb)
        # height=h.shape[2]
        # h=rearrange(h,'b c h w -> b (h w) c').contiguous()
        h=self.mid.attn_1(h)
        # h=rearrange(h,'b (h w) c -> b c h w ',h=height).contiguous()
        h=self.mid.block_2(h,temb)

        h=self.norm_out(h)
        h=nonlinearity(h)
        h=self.conv_out(h)

        return h

    def get_feature(self,x):
        temb=None

        hs=[self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h=self.down[i_level].block[i_block](hs[-1],temb)
                if len(self.down[i_level].attn)>0:
                    h=self.down[i_level].attn[i_block](h)
                    
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h=hs[-1]
        h=self.mid.block_1(h,temb)
        h=self.mid.attn_1(h)
        h=self.mid.block_2(h,temb)

        return h