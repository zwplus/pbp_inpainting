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
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers import DDIMScheduler,DDPMScheduler,PNDMScheduler
from autoencoder_kl import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available

from sd_unet.unet_2d_condition import UNet2DConditionModel as Unet
from appearce_net.unet_2d_condition import UNet2DConditionModel as Appearce_Unet
from style_encoder import (
    CLIP_Image_Extractor,
    CLIP_Proj
)
from control_net import ControlNetConditioningEmbedding

import wandb


from tiktok_dataset_2 import diffusion_dataset
from pytorch_lightning.loggers import WandbLogger

seed_everything(1024)

class People_Background(pl.LightningModule):
    def __init__(self,
                unet_config:Dict=None,
                pose_net_config:Dict=None,
                people_config:Dict=None,
                scheduler_path:str=None,
                vae_path:str=None,
                out_path='',
                image_size=(4,64,64),
                condition_rate=0.1,
                condition_guidance=5,
                warm_up=5000,
                learning_rate=0.0001,target_step=80000,
                enable_xformers_memory_efficient_attention=True,
                batch_size=64):
        super().__init__()
        self.save_hyperparameters()

        self.fid=FrechetInceptionDistance(normalize=True)
        self.ssim=SSIM(data_range=1.0)
        self.psnr=PSNR(data_range=1.0)
        self.lpips=LPIPS(net_type='vgg',normalize=True)

        self.init_model(unet_config,pose_net_config,people_config,vae_path,enable_xformers_memory_efficient_attention)
        self.train_scheduler=DDPMScheduler.from_pretrained(scheduler_path)
        self.test_scheduler=DDIMScheduler.from_pretrained(scheduler_path)
        self.test_scheduler.set_timesteps(50)

        self.condition_rate=condition_rate
        self.condition_guidance=condition_guidance
        self.warm_up=warm_up
        self.lr=learning_rate
        self.target_step=target_step
        

        self.out_path=out_path
        self.laten_shape=image_size
        self.save_img_num=0
        self.tr=transforms.ToPILImage()

    def init_model(self,unet_config,pose_net_config,people_config,vae_path:str=None,
                enable_xformers_memory_efficient_attention:bool=True):
        self.laten_model=AutoencoderKL.from_pretrained(vae_path)
        self.laten_model.eval()
        self.laten_model.requires_grad_(False)

        self.unet=Unet.from_pretrained(unet_config['ck_path'])
        self.AppearceNet=Appearce_Unet.from_pretrained('/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/appearce',
                                                    ignore_mismatched_sizes=True)

        new_in_channels=4
        with torch.no_grad():

            conv_new = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=self.AppearceNet.conv_in.out_channels, 
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new.weight)  
            conv_new.weight.data = conv_new.weight.data * 0.  

            conv_new.weight.data = self.AppearceNet.conv_in.weight.data[:,:new_in_channels+4]

            conv_new.bias.data = self.AppearceNet.conv_in.bias.data  

            self.AppearceNet.conv_in = conv_new  
            self.AppearceNet.config['in_channels'] = new_in_channels 

            conv_new_2 = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=self.unet.conv_in.out_channels, 
                kernel_size=3,
                padding=1,
            )

            torch.nn.init.kaiming_normal_(conv_new_2.weight)  
            conv_new_2.weight.data = conv_new_2.weight.data * 0.  
            conv_new_2.weight.data = self.unet.conv_in.weight.data[:,:new_in_channels+4]
            conv_new_2.bias.data = self.unet.conv_in.bias.data  
            
            self.unet.conv_in = conv_new_2  
            self.unet.config['in_channels'] = new_in_channels 
            

        self.clip=CLIP_Image_Extractor(**people_config['clip_image_extractor'])
        self.clip.eval()
        self.clip.requires_grad_(False)

        self.proj=CLIP_Proj(**people_config['clip_proj'])
        self.people_proj=CLIP_Proj(**people_config['clip_proj'])
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(**pose_net_config)

        # temp=[[320,320],[640,640],[1280,1280],[1280],[1280,1280,1280],[640,640,640],[320,320,320]]
        # self.cross_attn_linear_proj=nn.ModuleList()
        # for block in temp:
        #     self.cross_attn_linear_proj.append(
        #         nn.ModuleList([nn.Linear(input_ch,1024) for input_ch in block])
        #     )
            

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                print('start xformer')
                self.unet.enable_xformers_memory_efficient_attention()
                self.AppearceNet.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")

    def training_step(self, batch, batch_idx):
        self.save_img_num=0
        rate=random.random()

        background_img,people_vae,people_clip,back_clip,pose_img,img=batch

        img=img.to(torch.float16).to(self.device)
        people_vae=people_vae.to(torch.float16).to(self.device)
        people_clip=people_clip.to(torch.float16).to(self.device)
        back_clip=back_clip.to(torch.float16).to(self.device)
        background_img=background_img.to(torch.float16).to(self.device)
        pose_img=pose_img.to(torch.float16).to(self.device)
            
        pose_laten=self.get_control_cond(pose_img)
        background=self.img_to_laten(background_img)[0]
        back_clip=self.get_image_clip(back_clip)
        people_clip=self.get_people_clip(people_clip)
        people_laten=self.img_to_laten(people_vae)[0]
        

        if rate <= self.condition_rate:
            people_clip=torch.zeros_like(people_clip,dtype=torch.float16).to(self.device)
            pose_laten=torch.zeros_like(pose_laten,dtype=torch.float16).to(self.device)
            background=torch.zeros_like(background,dtype=torch.float16).to(self.device)


        target=self.img_to_laten(img)[0] 
        noise=torch.randn(target.shape,dtype=torch.float16).to(self.device)
        timesteps=torch.randint(0,self.train_scheduler.config.num_train_timesteps,(target.shape[0],)).long().to(self.device)
        noisy_image=self.train_scheduler.add_noise(target,noise,timesteps).to(torch.float16)
        
        people_laten=torch.cat([noisy_image,people_laten],dim=1)
        appearce_output=self.AppearceNet(sample=people_laten,timestep=timesteps,encoder_hidden_states=back_clip)
        self_attn_states=appearce_output.self_attn_states
        if rate <= self.condition_rate:
            self_attn_states=[[ torch.zeros_like(j_,dtype=torch.float16).to(self.device) for j_ in i_] for i_ in self_attn_states]

        laten=torch.cat([noisy_image,background],dim=1)
        model_out = self(laten,timesteps,self_attn_states,people_clip,pose_laten)

        loss=F.mse_loss(model_out,noise)
        self.log('train/loss',loss, prog_bar=True,
                    logger=True, on_step=True, on_epoch=True,sync_dist=True)
        self.log("global/step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('global/lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def forward(self,laten:torch.FloatTensor=None,timesteps:torch.Tensor=None,self_attn_states:torch.FloatTensor=None
                ,cross_attn_states:torch.FloatTensor=None,pose_laten:torch.FloatTensor=None): 
                return self.unet(sample=laten,timestep=timesteps,
                        encoder_hidden_states=cross_attn_states,
                        self_attn_states=self_attn_states,pose_laten=pose_laten).sample
    
    @torch.no_grad()
    def sample(self,people_vae:torch.FloatTensor,people_clip:torch.FloatTensor,back_clip:torch.FloatTensor,
                background_img:Optional[torch.FloatTensor]=None,pose_img:torch.FloatTensor=None,):

            latens_=torch.randn([people_vae.shape[0],*self.laten_shape],dtype=torch.float16).to(self.device)
            
            cond_pose_laten=self.get_control_cond(pose_img)
            uncond_pose_laten=torch.zeros_like(cond_pose_laten).to(torch.float16).to(self.device)
            pose_laten=torch.cat([cond_pose_laten,uncond_pose_laten])

            cond_people_clip=self.get_people_clip(people_clip)
            uncond_people_clip=torch.zeros_like(cond_people_clip).to(torch.float16).to(self.device)
            people_clip=torch.cat([cond_people_clip,uncond_people_clip])
            
            back_clip=self.get_image_clip(back_clip)
            cond_people_laten=self.img_to_laten(people_vae)[0]


            cond_back=self.img_to_laten(background_img)[0]
            back=torch.cat([cond_back,torch.zeros_like(cond_back,dtype=torch.float16).to(self.device)])

            
            for t in self.test_scheduler.timesteps:
                latens=torch.cat([latens_]*2)
                timestep=torch.full((latens.shape[0],),t).to(self.device)


                app_latens=torch.cat([latens_,cond_people_laten],dim=1)
                appearce_output=self.AppearceNet(sample=app_latens,
                                                timestep=timestep[:app_latens.shape[0]],
                                                encoder_hidden_states=back_clip)
                self_attn_states=[
                    [torch.cat([j_,torch.zeros_like(j_,dtype=torch.float16).to(self.device)]) for j_ in i_] for i_ in appearce_output.self_attn_states
                ]
                
                latens=torch.cat([latens,back],dim=1)
                
                noise_pred=self(latens,timestep,self_attn_states,people_clip,pose_laten)
                noise_cond,noise_uncond=noise_pred.chunk(2)
                noise_pred=noise_uncond+self.condition_guidance*(noise_cond-noise_uncond)
                latens_=self.test_scheduler.step(noise_pred,t,latens_).prev_sample
            return latens_

    @torch.no_grad()
    def validation_step(self,batch,batch_idx):

        background_img,people_vae,people_clip,back_clip,pose_img,img=batch
        img=img.to(torch.float16).to(self.device)
        people_vae=people_vae.to(torch.float16).to(self.device)
        people_clip=people_clip.to(torch.float16).to(self.device)
        back_clip=back_clip.to(torch.float16).to(self.device)
        background_img=background_img.to(torch.float16).to(self.device)
        pose_img=pose_img.to(torch.float16).to(self.device)
        
        target_img=self.sample(people_vae,people_clip,back_clip,background_img,pose_img)
        target_img=self.laten_to_img(target_img)
        target_img=torch.clamp(target_img.detach()/2+0.5,0,1).detach()
        img=(img.detach()/2+0.5).to(torch.float16)
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

    def get_image_clip(self,img_clip):
        with torch.no_grad():
            img_clip_feature=self.clip(img_clip).detach()
        img_clip_feature=self.proj(img_clip_feature)

        return img_clip_feature

    def get_people_clip(self,img_clip):
        with torch.no_grad():
            img_clip_feature=self.clip(img_clip).detach()
        img_clip_feature=self.people_proj(img_clip_feature)

        return img_clip_feature

    def get_control_cond(self,pose_img):
        pose_laten=self.controlnet_cond_embedding(pose_img)
        return pose_laten
    
    # def get_cross_attn(self,cross_attn_outputs:List):
    #     appearce_states=[]
    #     for attn_outputs,projs in zip(cross_attn_outputs,self.cross_attn_linear_proj):
    #         temp=[]
    #         for attn_output,proj in zip(attn_outputs,projs):
    #             temp.append(proj(attn_output))
    #         appearce_states.append(temp)
    #     return appearce_states
    
    
    def configure_optimizers(self):

        params =[i  for i in (list(self.people_proj.parameters())+list(self.unet.parameters())
                +list(self.AppearceNet.parameters())+list(self.controlnet_cond_embedding.parameters())
                +list(self.proj.parameters()))
                if i.requires_grad==True ]
        optim = torch.optim.AdamW(params, lr=self.lr)
        lambda_lr=lambda step: max(((self.global_step)/self.warm_up),5e-3) if (self.global_step)< self.warm_up else  1.0
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda_lr)
        return {'optimizer':optim,'lr_scheduler':{"scheduler":lr_scheduler,'monitor':'fid','interval':'step','frequency':1}}

    @torch.no_grad()
    def img_to_laten(self,imgs):
        encoder_hiddenstates=self.laten_model.encode(imgs)
        latens,last_feature=encoder_hiddenstates.latent_dist.sample().detach(),encoder_hiddenstates.last_feature.detach()
        latens=0.18215*latens
        return latens,last_feature
    
    @torch.no_grad()
    def laten_to_img(self,latens):
        latens=1/0.18215*latens
        return self.laten_model.decode(latens).sample.detach()


train_list=[
    '/data/zwplus/tiktok/train/train_new_pose.txt',
]
test_list=[
    '/data/zwplus/tiktok/test/test_new_pose.txt',
]




if __name__=='__main__':
    train_dataset=diffusion_dataset(train_list)
    test_dataset=diffusion_dataset(test_list,if_train=False)

    batch_size=20
    logger=WandbLogger(save_dir='/home/user/zwplus/pbp_inpainting/',project='pose_inpainting_ref')

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=32)
    val_loader=DataLoader(test_dataset,batch_size=batch_size,pin_memory=True,num_workers=32,drop_last=True)

    
    unet_config={
        'ck_path':'/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/unet-inpainting',
    }
    pose_net_config={
        'conditioning_embedding_channels': 320,
        'conditioning_channels':3,
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
            'local_num':8,
            'heads':8,
            'head_dim':128,
        }
    }


    vae_path='/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/vae'
    model=People_Background(unet_config,pose_net_config,people_config,scheduler_path='/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/scheduler',
                            vae_path=vae_path,out_path='/data/zwplus/pbp_inpainting/pose_inpainting_ref/output',condition_guidance=7.5,batch_size=batch_size,
                            warm_up=5000,learning_rate=1e-4)
    model.load_state_dict(torch.load('/data/zwplus/pbp_inpainting/pose_inpainting_ref/checkpoint/pndm-epoch=099-fid=35.156-ssim=0.669.ckpt/99.bin'),strict=False)
    # state_dict=torch.load('/data/zwplus/pbp_inpainting/pose_inpainting_ref/checkpoint/pndm-epoch=094-fid=34.875-ssim=0.666.ckpt/94_mask.ckpt')
    # state_dict['unet.conv_in.weight']=state_dict['unet.conv_in.weight'][:,:8,:,:]

    # logger.watch(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="/data/zwplus/pbp_inpainting/pose_inpainting_ref/checkpoint", 
                                                    save_top_k=5, monitor="fid",mode='min',
                                                    filename="pndm-{epoch:03d}-{fid:.3f}-{ssim:.3f}",)
    
    trainer=pl.Trainer(
        accelerator='gpu',devices=2,logger=logger,callbacks=[checkpoint_callback],
        default_root_dir='/data/zwplus/pbp_inpainting/pose_inpainting_ref/checkpoint',
        strategy=DeepSpeedStrategy(logging_level=logging.INFO,allgather_bucket_size=5e8,reduce_bucket_size=5e8),
        precision='16-mixed',  #bf16-mixed
        accumulate_grad_batches=16,check_val_every_n_epoch=5,
        log_every_n_steps=200,max_epochs=600,
        profiler='simple',benchmark=True,gradient_clip_val=1) 
    
    trainer.fit(model,train_loader,val_loader) 
    wandb.finish()

    # DeepSpeedStrategy(logging_level=logging.INFO,allgather_bucket_size=5e8,reduce_bucket_size=5e8)