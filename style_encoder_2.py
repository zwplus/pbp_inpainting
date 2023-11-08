import torch
import torch.nn as nn
import sys
sys.path.append('/home/user/zwplus/paper/')
from utils.attention import xformers_LinearCrossAttention
from transformers import CLIPVisionModelWithProjection
from einops import rearrange

class CLIP_Image_Extractor(nn.Module):
    def __init__(self,clip_path='/root/data1/github/pose_transfer/sd-image-variations-diffusers/image_encoder') -> None:
        super().__init__()
        self.clip_image_encoder=CLIPVisionModelWithProjection.from_pretrained(clip_path)
        self.clip_image_encoder.eval()
        self.clip_image_encoder.requires_grad_(False)

    def clip_encode_image_local(self, image): # clip local feature
        last_hidden_states = self.clip_image_encoder(image).last_hidden_state
        last_hidden_states_norm = self.clip_image_encoder.vision_model.post_layernorm(last_hidden_states)
        image_embedding=last_hidden_states_norm[:,0,:]
        return last_hidden_states_norm,image_embedding
    def forward(self,image):
        return self.clip_encode_image_local(image)

class CLIP_Proj(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,ck_path:str=None) -> None:
        super().__init__()
        self.refer_proj=nn.Linear(in_channel,out_channel,bias=False)
        # self.refer_proj.load_state_dict(torch.load(ck_path))
    
    def forward(self,last_hidden_states_norm,num_images_per_prompt=1):
        image_embeddings = self.refer_proj(last_hidden_states_norm)
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return image_embeddings     


class FeedFoward(nn.Module):
    def __init__(self,in_channels:int=768,mult:int=2 ) -> None:
        super().__init__()
        self.proj_in=nn.Linear(in_channels,in_channels*2)
        self.act_fn=nn.GELU()
        self.proj_out=nn.Linear(in_channels*mult,in_channels)
    def forward(self,x):
        x=self.proj_in(x)
        x=self.act_fn(x)
        x=self.proj_out(x)
        return x


class clip_transformer_block(nn.Module):
    def __init__(self,inchannels:int=1024,mult=2,heads_num=16,head_dim=64,cross_dim:int=None,**kwargs) -> None:
        super().__init__()
        cross_dim=inchannels if cross_dim is None else cross_dim
        inner_dim=heads_num*head_dim

        self.norm_in=nn.LayerNorm(inchannels)
        self.proj_in=nn.Linear(inchannels,inner_dim)

        self.norm_1=nn.LayerNorm(inner_dim)
        self.attn1=xformers_LinearCrossAttention(query_dim=inner_dim,context_dim=cross_dim,heads=heads_num,dim_head=head_dim)
        
        self.norm_2=nn.LayerNorm(inner_dim)
        self.feedforward=FeedFoward(inner_dim,mult=mult)

        self.norm_out=nn.LayerNorm(inner_dim)
        self.proj_out=nn.Linear(inner_dim,inchannels)
    
    def forward(self,part_feature:torch.FloatTensor,full_feature:torch.FloatTensor=None):
        
        res=part_feature
        part_feature=self.norm_in(part_feature)
        part_feature=self.proj_in(part_feature)

        part_feature=self.norm_1(part_feature)
        residual=part_feature
        part_feature=self.attn1(part_feature,full_feature)
        part_feature+=residual

        part_feature=self.norm_2(part_feature)
        part_feature=self.feedforward(part_feature)


        part_feature=self.norm_out(part_feature)
        part_feature=self.proj_out(part_feature)+res

        return part_feature



class people_global_fusion(nn.Module):
    def __init__(self,inchannels=512,ch=768,local_num=8,heads=8) -> None:
        super().__init__()
        self.local_num=local_num
        self.conv=nn.Conv2d(inchannels,ch,kernel_size=1)
        self.norm=nn.GroupNorm(num_groups=32,num_channels=inchannels)
        self.silu=nn.SiLU()
        self.attn=xformers_LinearCrossAttention(ch,ch,heads=8,dim_head=ch//heads)
    
    def forward(self,x,full_people_feature):
        B,C,H,W=x.shape
        x=self.norm(x)
        x=self.silu(x)
        x=self.conv(x)
        h=rearrange(x,'(b l) c h w -> b (l h w) c',l=self.local_num).contiguous()
        h=self.attn(h,full_people_feature)
        h=rearrange(h,'b (l h w) c -> (b l) c h w',l=self.local_num,h=H).contiguous()

        return h+x

class people_local_fusion(nn.Module):
    def __init__(self,inchannels=768,mult=2,local_num=8,heads_num=8,head_dim=128,eps=1e-5,**kwargs) -> None:
        super().__init__()

        self.inchannels=inchannels
        self.local_num=local_num

        self.norm=nn.GroupNorm(32,self.inchannels,eps=1e-5)
        self.silu=nn.SiLU()
        self.conv=nn.Conv2d(self.inchannels,self.inchannels,5,1,2)

        inner_dim=head_dim*heads_num
        self.proj_in=nn.Linear(self.inchannels,inner_dim)

        self.attn1=xformers_LinearCrossAttention(inner_dim,heads=heads_num,dim_head=head_dim)    
        self.attn2=xformers_LinearCrossAttention(inner_dim,heads=heads_num,dim_head=head_dim)   
        self.feedfoward=nn.Sequential(
            nn.Linear(inner_dim,inner_dim*mult),
            nn.GELU(),
            nn.Linear(inner_dim*mult,inner_dim)
        )

        self.norm1=nn.LayerNorm(inner_dim)
        self.norm2=nn.LayerNorm(inner_dim)
        self.norm3=nn.LayerNorm(inner_dim)

        self.proj_out=nn.Linear(inner_dim,self.inchannels)

                
    
    def forward(self,x):
        '''
            x=(b l) c h w
        '''
        x=self.norm(x)
        x=self.silu(x)
        h=self.conv(x)
        batch,ch,height,width=x.shape
        h=rearrange(h,'b c h w -> b c (h w)').contiguous()
        h=torch.softmax(h,dim=-1)
        h=rearrange(h,'b c (h w) -> b c h w',h=height).contiguous()
        h=torch.sum(x*h,dim=[2,3])
        
        h=rearrange(h,'(b l) c -> b l c',l=self.local_num).contiguous()
        resdiual=h

        h=self.proj_in(h)
        h=self.attn1(self.norm1(h))+h
        h=self.attn2(self.norm2(h))+h
        out=self.feedfoward(self.norm3(h))+h
        out=self.proj_out(h)

        return out+resdiual

if __name__ == '__main__':
    net=people_global_fusion()
    
    x=torch.randn((1,1,1024))
    t=torch.randn((1,1,768))
    from torch.optim import AdamW
    opt=AdamW(list(net.parameters()),lr=0.0001)



    opt.zero_grad()

    p=net(x)
    loss=torch.nn.functional.l1_loss(p,t)
    loss.backward()
    for name, param in net.named_parameters():
        if param.grad is None:
            print(name)
    opt.step()