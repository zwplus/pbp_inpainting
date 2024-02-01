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
        self.data_dict={}
        self.if_train=if_train
        pairs_list=[]
        for i in data_pairs_txt_list:
            with open(i,'r') as f:
                pairs_list.extend(f.readlines())
        pairs_list.sort()
        
        for i in pairs_list:
            i=i.strip()
            img_path,mask_img_path,pose_img_path,people_img_path,back_img_path=i.split(',')

            file_dir=img_path.split('/')[-2]
            file_name=img_path.split('/')[-1]
            file_name=file_name.split('.')[0]+'.png'

            count=0
            if not ( os.path.isfile(img_path) and os.path.isfile(pose_img_path)
                    and os.path.isfile(mask_img_path)):
                print(img_path)
                print(mask_img_path)
                print(pose_img_path)
                count+=1
            else:
                self.data_pairs.append((img_path,mask_img_path,pose_img_path,people_img_path,back_img_path))
                if file_dir not in self.data_dict.keys():
                    self.data_dict[file_dir]=[]
                self.data_dict[file_dir].append((people_img_path,back_img_path,mask_img_path))
        print(count)
        
        self.random_square_height = transforms.Lambda(lambda img: transforms.functional.crop(img, top=int(torch.randint(0, img.height - img.width, (1,)).item()), left=0, height=img.width, width=img.width))
        self.random_square_width = transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=int(torch.randint(0, img.width - img.height, (1,)).item()), height=img.height, width=img.height))

        min_crop_scale = 0.8 if if_train else 1.0
        
        print(len(pairs_list))
        self.transformer_ae=transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (512,512),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.cond_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (512,512),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transformer_mask=transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (512,512),
                scale=(min_crop_scale, 1.0), ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize((64,64),interpolation=transforms.InterpolationMode.BICUBIC)
            ]
        )

        self.transformer_clip=CLIPImageProcessor.from_pretrained('/home/user/zwplus/pbp_inpainting/sd-2.1/feature_exract')
        
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
            raw,_,pose,_,_=self.data_pairs[index]
            file_dir=raw.split('/')[-2]
            pose=Image.open(pose)
            raw=Image.open(raw)
            if self.if_train==False:
                people,back,back_mask=self.data_dict[file_dir][0]
                people=Image.open(people)
                back=Image.open(back)
                back_mask=Image.open(back_mask)
                back_mask =back_mask.convert("L") 
                dense_mask=pose.convert("L")

                transform1=None
            else:
                back_index=random.randint(0,len(self.data_dict[file_dir])-1)
                people,back,back_mask=self.data_dict[file_dir][back_index]
                back=Image.open(back)

                back_mask=Image.open(back_mask)
                back_mask = back_mask.convert("L") 
                dense_mask=pose.convert("L")

                # people_index=random.randint(0,len(self.data_dict[file_dir])-1)
                people=Image.open(people)

                if raw.size[0]>raw.size[1]:  # w>h
                    transform1=self.random_square_width
                elif raw.size[0]<raw.size[1]: #h>w:
                    transform1=self.random_square_height
                else:
                    transform1=None
            
            
            state = torch.get_rng_state()

            raw=self.augmentation(raw, transform1, self.transformer_ae, state)
            pose=self.augmentation(pose, transform1, self.cond_transform, state)


            people_vae=self.augmentation(people,transform1,self.transformer_ae,state)
            people_clip=self.augmentation(people,transform1,self.transformer_clip,state).pixel_values[0]
            
            back_vae=self.augmentation(back,transform1,self.transformer_ae,state)
            back_clip=self.augmentation(back,transform1,self.transformer_clip,state).pixel_values[0]
            back_mask=self.augmentation(back_mask,transform1,self.transformer_mask,state)
            back_mask[back_mask>=0.5]=1
            back_mask[back_mask<0]=0

            dense_mask=self.augmentation(dense_mask, transform1, self.transformer_mask, state)
            dense_mask[dense_mask>=0.5]=1
            dense_mask[dense_mask<0]=0
            
        except Exception as e:
            print(file_dir)
            traceback.print_exc()
        return back_vae,people_vae,people_clip,back_clip,pose,raw,dense_mask,back_mask

train_list=[
    '/data/zwplus/tiktok_image/test_list.txt',
]
t=diffusion_dataset(train_list)
next(iter(t))