import sys
import os
from typing import Any, Dict,List,Optional
import random
import logging
log = logging.getLogger(__name__)

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms

from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers import AutoencoderKL,PNDMScheduler,DDIMScheduler,DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

from unet_2d_condition import UNet2DConditionModel as Unet
from style_encoder_2 import (
    CLIP_Image_Extractor,
    CLIP_Proj,
    clip_transformer_block
)

import wandb

from tiktok_dataset import diffusion_dataset
from control_net import ControlNetModel
from pytorch_lightning.loggers import WandbLogger

seed_everything(24)

class People_Background(pl.LightningModule):
    def __init__(self,
                unet_config:Dict=None,
                people_config:Dict=None,
                scheduler_path:str=None,
                vae_path:str=None,
                out_path='',
                image_size=(4,32,32),condition_rate=0.1,condition_guidance=2,
                warm_up=3000,learning_rate=0.0001,target_step=300000,
                local_num=8,enable_xformers_memory_efficient_attention=True,batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.init_model(unet_config,people_config,vae_path,enable_xformers_memory_efficient_attention)
        self.train_scheduler=DDPMScheduler.from_pretrained(scheduler_path)
        self.test_scheduler=DDIMScheduler.from_pretrained(scheduler_path)
        self.test_scheduler.set_timesteps(50)

        self.condition_rate=condition_rate
        self.condition_guidance=condition_guidance
        self.warm_up=warm_up
        self.lr=learning_rate
        self.target_step=target_step
        self.local_num=local_num
        

        self.out_path=out_path
        self.laten_shape=image_size
        self.save_img_num=0
        self.tr=transforms.ToPILImage()

    def init_model(self,unet_config,people_config,vae_path:str=None,
                enable_xformers_memory_efficient_attention:bool=True):
        self.laten_model=AutoencoderKL.from_pretrained(vae_path)
        self.laten_model.eval()
        self.laten_model.requires_grad_(False)


        self.unet=Unet.from_pretrained(unet_config['ck_path'])
        new_in_channels=8
        with torch.no_grad():

            conv_new = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=self.unet.conv_in.out_channels, 
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new.weight)  
            conv_new.weight.data = conv_new.weight.data * 0.  

            conv_new.weight.data = self.unet.conv_in.weight.data[:, :8]  
            conv_new.bias.data = self.unet.conv_in.bias.data  

            self.unet.conv_in = conv_new  
            self.unet.config['in_channels'] = new_in_channels 

        self.clip=CLIP_Image_Extractor(**people_config['clip_image_extractor'])
        self.clip.eval()
        self.clip.requires_grad_(False)
        
        self.people_proj=CLIP_Proj(**people_config['clip_proj'])
        self.people_local_fusion=clip_transformer_block(**people_config['local_fusion'])

        self.controlnet_pose = ControlNetModel.from_unet(unet=self.unet)


        self.fid=FrechetInceptionDistance(normalize=True)
        self.ssim=SSIM(data_range=1.0)
        self.psnr=PSNR(data_range=1.0)
        self.lpips=LPIPS(net_type='vgg',normalize=True)

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                print('start xformer')
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet_pose.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")

    def training_step(self, batch, batch_idx):
        self.save_img_num=0
        rate=random.random()

        background_img,part_img,pose_img,img=batch
        img=img.to(torch.bfloat16).to(self.device)
        part_img=part_img.to(torch.bfloat16).to(self.device)
        background_img=background_img.to(torch.bfloat16).to(self.device)
        pose_img=pose_img.to(torch.bfloat16).to(self.device)
            

        if rate <= self.condition_rate:
            people_feature=self.get_people_condition(
                torch.zeros_like(part_img).to(torch.bfloat16).to(self.device))
            background=self.img_to_laten(torch.zeros_like(background_img).to(torch.bfloat16).to(self.device))
        else:
            background=self.img_to_laten(background_img)
        
        target=self.img_to_laten(img)
        

        noise=torch.randn(target.shape,dtype=torch.bfloat16).to(self.device)
        timesteps=torch.randint(0,self.train_scheduler.config.num_train_timesteps,(target.shape[0],)).long().to(self.device)
        noisy_image=self.train_scheduler.add_noise(target,noise,timesteps).to(torch.bfloat16)
        
        laten=torch.cat([noisy_image,background],dim=1)
        model_out = self(laten, timesteps,people_feature,pose_img)

        loss=F.mse_loss(model_out,noise)

        self.log('train/loss',loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("global/step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('global/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def forward(self,laten:torch.FloatTensor=None,timesteps:torch.Tensor=None,
                people_feature:torch.FloatTensor=None,pose_img:torch.FloatTensor=None): 
                down_block_res_samples, mid_block_res_sample=self.controlnet_pose(
                        sample=laten,timestep=timesteps,
                        encoder_hidden_states=people_feature, # both controlnet path use the refer latents
                        controlnet_cond=pose_img, conditioning_scale=1.0, return_dict=False)
                return self.unet(sample=laten,timestep=timesteps,
                        encoder_hidden_states=people_feature,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,).sample
    
    @torch.no_grad()
    def sample(self,part_img:torch.FloatTensor,
                background_img:Optional[torch.FloatTensor]=None,pose_img:torch.FloatTensor=None):

            latens_=torch.randn([background_img.shape[0],*self.laten_shape],dtype=torch.bfloat16).to(self.device)

            uncond_people_feature=self.get_people_condition(torch.zeros_like(part_img).to(torch.bfloat16).to(self.device))
            cond_people_feature=self.get_people_condition(part_img)
            people_feature=torch.cat([cond_people_feature,uncond_people_feature])

            uncond_back=self.img_to_laten(torch.zeros_like(background_img).to(torch.bfloat16).to(self.device))
            cond_back=self.img_to_laten(background_img).detach()
            back=torch.cat([cond_back,uncond_back])
            
            pose_img=torch.cat([pose_img,pose_img])

            for t in self.test_scheduler.timesteps:
                latens=torch.cat([latens_]*2)
                latens=torch.cat([latens,back],dim=1)
                timestep=torch.full((latens.shape[0],),t).to(self.device)
                noise_pred=self(latens,timestep,people_feature,pose_img)
                noise_cond,noise_uncond=noise_pred.chunk(2)
                noise_pred=noise_uncond+self.condition_guidance*(noise_cond-noise_uncond)
                latens_=self.test_scheduler.step(noise_pred,t,latens_).prev_sample
            return latens_

    @torch.no_grad()
    def validation_step(self,batch,batch_idx):

        background_img,part_img,pose_img,img=batch
        img=img.to(torch.bfloat16).to(self.device)
        part_img=part_img.to(torch.bfloat16).to(self.device)
        background_img=background_img.to(torch.bfloat16).to(self.device)
        pose_img=pose_img.to(torch.bfloat16).to(self.device)
        
        target_img=self.sample(part_img,background_img,pose_img)
        target_img=self.laten_to_img(target_img)
        target_img=torch.clamp(target_img.detach()/2+0.5,0,1).detach()
        img=(img.detach()/2+0.5).to(torch.bfloat16)
        pose_img=pose_img.detach().cpu()/2+0.5
            

        self.fid.update(target_img,real=False)
        self.fid.update(img,real=True)
        self.log('fid',self.fid,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.lpips.update(target_img,img)
        self.log('lpips',self.lpips,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.psnr.update(target_img,img)
        self.log('psnr',self.psnr,prog_bar=True,logger=True,on_step=True,on_epoch=True)

        self.ssim.update(target_img,img)
        self.log('ssim',self.ssim,prog_bar=True,logger=True,on_step=True,on_epoch=True)


        img=img.detach().cpu()
        pose_img=pose_img.detach().cpu()
        target_img=target_img.detach().cpu()

        file_dir=os.path.join(self.out_path,str(self.global_step))
        os.makedirs(file_dir,exist_ok=True)
        for i in range(target_img.shape[0]):
            save_img=self.tr(target_img[i])
            save_img.save(os.path.join(file_dir,str(self.local_rank)+'_'+str(self.save_img_num)+'.jpg'))
            self.save_img_num+=1
            if self.save_img_num==1:  
                h=torch.cat([img[:4],pose_img[:4],target_img[:4]])
                show_img=make_grid(h,nrow=4,padding=1)
                show_img=self.tr(show_img)

                logger.log_image(f'val/image',images=[show_img],step=self.global_step)

    def get_people_condition(self,part_img):
        part_img=rearrange(part_img,'b (l c) h w -> (b l) c h w',c=3).contiguous()

        part_laten=self.clip(part_img)[1].detach().unsqueeze(dim=1)
        part_laten=self.people_proj(part_laten)
        part_laten=rearrange(part_laten,'(b l) n d -> b (l n) d',l=self.local_num)

        people_local_feature=self.people_local_fusion(part_laten)

        return people_local_feature

    
    def configure_optimizers(self):

        params =[i  for i in (list(self.people_proj.parameters())
                +list(self.people_local_fusion.parameters())+list(self.unet.parameters())
                +list(self.controlnet_pose.parameters()))
                    if i.requires_grad==True ]
        optim = torch.optim.AdamW(params, lr=self.lr)
        lambda_lr=lambda step: max(((self.global_step)/self.warm_up),5e-3) if (self.global_step)< self.warm_up else  \
                                                                        max((self.target_step-self.global_step)/(self.target_step-self.warm_up),1e-3)
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda_lr)
        return {'optimizer':optim,'lr_scheduler':{"scheduler":lr_scheduler,'monitor':'fid','interval':'step','frequency':1}}

    @torch.no_grad()
    def img_to_laten(self,imgs):
        latens=self.laten_model.encode(imgs).latent_dist.sample().detach()
        latens=0.18215*latens
        return latens
    
    @torch.no_grad()
    def laten_to_img(self,latens):
        latens=1/0.18215*latens
        return self.laten_model.decode(latens).sample.detach()


train_list=[
    '/data/zwplus/tiktok/train/titok_pairs.txt',
]
test_list=[
    '/data/zwplus/tiktok/test/titok_pairs.txt'
]


if __name__=='__main__':
    train_dataset=diffusion_dataset(train_list)
    test_dataset=diffusion_dataset(test_list,if_train=False)

    batch_size=32
    logger=WandbLogger(save_dir='/home/user/zwplus/pbp_inpainting/',project='pose_inpainting_A800')

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=32)
    val_loader=DataLoader(test_dataset,batch_size=batch_size,pin_memory=True,num_workers=32,drop_last=True)

    
    unet_config={
        'ck_path':'/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/unet-inpainting',
    }
    people_config={
        'clip_image_extractor':
            {
                'clip_path':'/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/image'
            },
        'clip_proj':{
            'in_channel':1280,
            'out_channel':1024,
            'ck_path':'/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/proj/proj.bin'
        },
        'global_fusion':{
            'inchannels':512,
            'ch':1024,
            'local_num':8,
            'heads':8,
        },
        'local_fusion':{
            'inchannels':1024,
            'mult':2,
            'heads_num':16,
            'head_dim':64,
        }
    }


    vae_path='/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/vae'
    model=People_Background(unet_config,people_config,scheduler_path='/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/scheduler',
                            vae_path=vae_path,out_path='/home/user/zwplus/pbp_inpainting/output',learning_rate=8e-5
                            ,warm_up=10000,batch_size=32)
    
    logger.watch(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="/home/user/zwplus/pbp_inpainting/checkpoint", 
                                                    save_top_k=3, monitor="fid",mode='min',
                                                    filename="pndm-{epoch:03d}-{fid:.3f}-{ssim:.3f}",)
    
    trainer=pl.Trainer(
        accelerator='gpu',devices=2,logger=logger,callbacks=[checkpoint_callback],
        default_root_dir='/home/user/zwplus/pbp_inpainting/checkpoint',
        strategy=DeepSpeedStrategy(allgather_bucket_size=5e8,reduce_bucket_size=5e8)
        ,precision='bf16-mixed',  #bf16-mixed
        accumulate_grad_batches=8,check_val_every_n_epoch=10,
        log_every_n_steps=200,max_epochs=600,
        profiler='simple',benchmark=True,gradient_clip_val=1) 
    
    trainer.fit(model,train_loader,val_loader) 
    wandb.finish()