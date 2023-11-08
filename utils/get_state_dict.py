import pytorch_lightning as pl
import torch


ck=torch.load('/root/data1/github/pose_transfer/checkpoint/diffusion/ddpm-epoch=023-fid=25.032.ckpt')
style_encoder_dict={}
print(ck['state_dict'].keys())
for i in ck['state_dict'].keys():
    if 'style_encoder' == i.split('.')[0]:
        style_encoder_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]
fusion_dict={}
for i in ck['state_dict'].keys():
    if 'fusion' == i.split('.')[0]:
        fusion_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]     
        
model_dict={}
for i in ck['state_dict'].keys():
    if 'model' == i.split('.')[0]:
        model_dict['.'.join(i.split('.')[1:])]=ck['state_dict'][i]

torch.save(style_encoder_dict,'/root/data1/github/pose_transfer/checkpoint/diffusion/style_encoder_dict_2.ck')   
torch.save(fusion_dict,'/root/data1/github/pose_transfer/checkpoint/diffusion/fusion_2.ck')
torch.save(model_dict,'/root/data1/github/pose_transfer/checkpoint/diffusion/unet_2.ck')