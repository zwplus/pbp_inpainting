import torch
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms
import random
import json
import cv2
from transformers import CLIPImageProcessor
import traceback

class diffusion_dataset(Dataset):
    def __init__(self,data_pairs_txt_list,if_train=True) -> None:
        super().__init__()
        self.data_pairs=[]
        pairs_list=[]
        for i in data_pairs_txt_list:
            with open(i,'r') as f:
                pairs_list.extend(f.readlines())
        
        for i in pairs_list:
            i=i.strip()
            target_img_path,people_img_path,back_image_path,pose_img,src_mask_img=i.split(',')
            
            
            if not ( os.path.isfile(people_img_path) and os.path.isfile(pose_img)
                    and os.path.isfile(back_image_path) and os.path.isfile(target_img_path) and os.path.isfile(src_mask_img) ):
                print(people_img_path)
            else:
                self.data_pairs.append((target_img_path,people_img_path,back_image_path,pose_img,src_mask_img))
        
        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        min_crop_scale = 0.8 if if_train else 1.0
        
        print(len(pairs_list))
        self.transformer_ae=transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (256,256),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.cond_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (256,256),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transformer_clip=CLIPImageProcessor.from_pretrained('/home/user/zwplus/pbp_inpainting/sd-2.1/feature_exract')
        
        self.transformer_mask=transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (256,256),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(
                (32,32),
                interpolation=transforms.InterpolationMode.BILINEAR),]
        )
    def __len__(self,):
        return len(self.data_pairs)

    def augmentation(self, frame, transform1, transform2=None, state=None):
        if state is not None:
            torch.set_rng_state(state)   #确保每次产生的随机数是相同的
        if  transform1 is not None:
            frame_transform1 = transform1(frame) 
        else: 
            frame_transform1=frame
        
        if transform2 is None:
            return frame_transform1
        else:
            return transform2(frame_transform1)

    def __getitem__(self, index):
        try:
            raw,people,back,pose,src_mask=self.data_pairs[index]
            back=Image.open(back)
            # pose=Image.open(pose)
            raw=Image.open(raw)
            people=Image.open(people)
            src_mask=Image.open(src_mask)
            
            if raw.size[0]>raw.size[1]:  # w>h
                transform1=self.random_square_width
            elif raw.size[0]<raw.size[1]: #h>w:
                transform1=self.random_square_height
            else:
                transform1=None
            
            state = torch.get_rng_state()

            raw=self.augmentation(raw, transform1, self.transformer_ae, state)
            # target_pose_cond=self.augmentation(pose, transform1, self.cond_transform, state)
            people_vae=self.augmentation(people,transform1,self.transformer_ae,state)
            people_clip=self.augmentation(people,transform1,self.transformer_clip,state).pixel_values[0]
            back_vae=self.augmentation(back,transform1,self.transformer_ae,state)

            src_mask=self.augmentation(src_mask,transform1,self.transformer_mask,state)
            src_mask[src_mask>=0.5]=1
            src_mask[src_mask<0.5]=0
            
        except Exception as e:
            print(pose)
            traceback.print_exc()
        return back_vae,people_vae,people_clip,raw,src_mask