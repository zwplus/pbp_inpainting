import math
import torch
import torch.nn.functional as F
from einops import rearrange,repeat
from torch import nn,einsum

from flash_attn import  flash_attn_func,flash_attn_qkvpacked_func
from xformers.ops import memory_efficient_attention

def Normalize(in_channels):
    return nn.GroupNorm(32,in_channels,eps=1e-6,affine=True)

class LinearAttention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64) -> None:
        super().__init__()
        self.heads=heads
        self.scale=dim_head**-0.5
        hidden_dim=dim_head*self.heads
        self.to_qkv=nn.Conv2d(dim,hidden_dim*3,kernel_size=1,bias=False)
        self.to_out=nn.Conv2d(hidden_dim,dim,1)
    
    def forward(self,x):
        b,c,h,w=x.shape
        qkv=self.to_qkv(x)
        qkv=rearrange(qkv,'b (qkv heads c) h w -> b (h w) qkv heads c',heads=self.heads,qkv=3)
        context=flash_attn_qkvpacked_func(qkv)
        out=rearrange(context,'b (h w) heads c -> b (heads c) h w',h=h,w=w)
        return self.to_out(out)

class LinearSelfAttention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64) -> None:
        super().__init__()
        self.heads=heads
        hidden_dim=dim_head*self.heads
        self.to_qkv=nn.Linear(dim,hidden_dim*3)
        self.to_out=nn.Linear(hidden_dim,dim)
    
    def forward(self,x):
        b,c,h,w=x.shape
        x=rearrange(x,'b c h w -> b (h w) c').contiguous()
        qkv=self.to_qkv(x)
        qkv=rearrange(qkv,'b l (qkv heads c) -> b l qkv heads c',heads=self.heads,qkv=3)
        context=flash_attn_qkvpacked_func(qkv)
        out=rearrange(context,'b l heads c -> b l (heads c)')
        out=self.to_out(out)+x
        out=rearrange(out,'b (h w) c -> b c h w',h=h).contiguous()
        return out 

class flash_SpatialSelfAttention(nn.Module):
    def __init__(self,in_channels,heads=8,dim_head=64) -> None:
        super().__init__()

        self.heads=heads
        hidden_dim=dim_head*self.heads
        self.norm=Normalize(in_channels)
        self.qkv=torch.nn.Conv2d(in_channels,hidden_dim*3,kernel_size=1,stride=1,padding=0)
        self.proj_out=torch.nn.Conv2d(hidden_dim,in_channels,kernel_size=1,stride=1,padding=0)
    
    def forward(self,x):
        b,c,h,w=x.shape
        h_=self.norm(x)
        qkv=self.qkv(h_)
        qkv=rearrange(qkv,'b (qkv heads c) h w -> b (h w) qkv heads c',heads=self.heads,qkv=3)
        context=flash_attn_qkvpacked_func(qkv)
        out=rearrange(context,'b (h w) heads c -> b (heads c) h w',h=h)
        out=self.proj_out(out)
        out=x+out
        return out



class SpatialSelfAttention(nn.Module):
    def __init__(self,in_channels,heads=8,dim_head=64) -> None:
        super().__init__()

        self.in_channels=in_channels
        self.heads=heads
        hidden_dim=dim_head*self.heads
        self.norm=Normalize(in_channels)

        self.q=torch.nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.k=torch.nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.v=torch.nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)

        self.proj_out=torch.nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
    
    def forward(self,x):
        h_=x
        h_=self.norm(h_)
        q=self.q(h_)
        k=self.k(h_)
        v=self.v(h_)

        b,c,h,w=q.shape
        q=rearrange(q,'b c h w -> b (h w) c')
        k=rearrange(k,'b c h w -> b c (h w)') 
        w_=torch.einsum('bij,bjk->bik',q,k)
        w_=w_*(int(c)**(-0.5))
        w_=torch.nn.functional.softmax(w_,dim=-1)
        v=rearrange(v,'b c h w -> b c (h w)')
        w_=rearrange(w_,'b i j -> b j i')
        h_=torch.einsum('bij,bjk->bik',v,w_)
        h_=rearrange(h_,'b c (h w) -> b c h w',h=h)
        h_=self.proj_out(h_)
        r=x+h_
        return r

#二维的交叉注意力机制
class flash_SpatialCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.) -> None:
        super().__init__()

        inner_dim=dim_head*heads
        context_dim=query_dim if context_dim is None else context_dim

        self.norm=Normalize(query_dim)
        self.heads=heads
        
        self.to_q=nn.Conv2d(query_dim,inner_dim,1,1,0,bias=False)
        self.to_k=nn.Conv2d(context_dim,inner_dim,1,1,0,bias=False)
        self.to_v=nn.Conv2d(context_dim,inner_dim,1,1,0,bias=False)

        self.to_out=nn.Sequential(
            nn.Conv2d(inner_dim,query_dim,1,1,0,bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self,x,context=None):
        heads=self.heads
        b,c,h,w=x.shape
        x=self.norm(x)
        context=x if context is None else context

        q=self.to_q(x)
        q=rearrange(q,'b (heads d) h w -> b (h w) heads d ',heads=heads)
        k=self.to_k(context)
        k=rearrange(k,'b (heads d) h w -> b (h w) heads d ',heads=heads)
        v=self.to_v(context)
        v=rearrange(v,'b (heads d) h w -> b (h w) heads d ',heads=heads)

        out=flash_attn_func(q,k,v)
        out=rearrange(out,'b (h w) heads d -> b (heads d) h w',heads=heads,h=h,w=w)
        return self.to_out(out)

class xformers_SpatialCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.) -> None:
        super().__init__()

        inner_dim=dim_head*heads
        context_dim=query_dim if context_dim is None else context_dim

        self.norm=Normalize(query_dim)
        self.heads=heads
        
        self.to_q=nn.Conv2d(query_dim,inner_dim,1,1,0,bias=False)
        self.to_k=nn.Conv2d(context_dim,inner_dim,1,1,0,bias=False)
        self.to_v=nn.Conv2d(context_dim,inner_dim,1,1,0,bias=False)

        self.to_out=nn.Sequential(
            nn.Conv2d(inner_dim,query_dim,1,1,0,bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self,x,context=None):
        heads=self.heads
        b,c,h,w=x.shape
        x=self.norm(x)
        context=x if context is None else context

        q=self.to_q(x)
        q=rearrange(q,'b (heads d) h w -> b (h w) heads d ',heads=heads)
        k=self.to_k(context)
        k=rearrange(k,'b (heads d) h w -> b (h w) heads d ',heads=heads)
        v=self.to_v(context)
        v=rearrange(v,'b (heads d) h w -> b (h w) heads d ',heads=heads)

        out=memory_efficient_attention(q,k,v)
        out=rearrange(out,'b (h w) heads d -> b (heads d) h w',heads=heads,h=h,w=w)
        return self.to_out(out)

class LinearCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.) -> None:
        super().__init__()

        inner_dim=dim_head*heads
        context_dim=query_dim if context_dim is None else context_dim

        self.scale=dim_head**-0.5
        self.heads=heads
        
        self.to_q=nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k=nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v=nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,query_dim,bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self,x,context=None):
        '''
        x:b (h,w) c
        context : b embed c
        '''
        
        if context is None:
            context=x
        q=self.to_q(x)
        k=self.to_k(context)
        v=self.to_v(context)

        q=rearrange(q,'b i (h c) -> (b h) i c',h=self.heads)
        k=rearrange(k,'b i (h c) -> (b h) i c',h=self.heads)
        v=rearrange(v,'b i (h c) -> (b h) i c',h=self.heads)

        query=torch.einsum('b i c,b j c->b i j',q,k)*self.scale
        query=torch.softmax(query,dim=-1)
        value=torch.einsum('b i j,b j c -> b i c',query,v)
        out=rearrange(value,'(b h) i c -> b i (h c)',h=self.heads)
        return self.to_out(out)

class flash_LinearCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.) -> None:
        super().__init__()

        inner_dim=dim_head*heads
        context_dim=query_dim if context_dim is None else context_dim
        self.heads=heads
        
        self.to_q=nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k=nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v=nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,query_dim,bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self,x,context=None):
        '''
        x:b (h,w) c
        context : b embed c
        '''    
        if context is None:
            context=x
        q=self.to_q(x)
        q=rearrange(q,'b i (h c) -> b i h c ',h=self.heads)
        k=self.to_k(context)
        k=rearrange(k,'b i (h c) -> b i h c ',h=self.heads)
        v=self.to_v(context)
        v=rearrange(v,'b i (h c) -> b i h c ',h=self.heads)
        value=flash_attn_func(q,k,v)
        out=rearrange(value,'b i h c -> b i (h c)',h=self.heads)
        return self.to_out(out)


class xformers_LinearCrossAttention(nn.Module):
    def __init__(self,query_dim,context_dim=None,heads=8,dim_head=64,dropout=0.) -> None:
        super().__init__()

        inner_dim=dim_head*heads
        context_dim=query_dim if context_dim is None else context_dim
        self.heads=heads
        
        self.to_q=nn.Linear(query_dim,inner_dim,bias=False)
        self.to_k=nn.Linear(context_dim,inner_dim,bias=False)
        self.to_v=nn.Linear(context_dim,inner_dim,bias=False)

        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,query_dim,bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self,x,context=None):
        '''
        x:b (h,w) c
        context : b embed c
        '''    
        if context is None:
            context=x
        q=self.to_q(x)
        q=rearrange(q,'b i (h c) -> b i h c ',h=self.heads)
        k=self.to_k(context)
        k=rearrange(k,'b i (h c) -> b i h c ',h=self.heads)
        v=self.to_v(context)
        v=rearrange(v,'b i (h c) -> b i h c ',h=self.heads)
        value=memory_efficient_attention(q,k,v)
        out=rearrange(value,'b i h c -> b i (h c)',h=self.heads)
        return self.to_out(out)









        

        



