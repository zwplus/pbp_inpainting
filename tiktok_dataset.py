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
            target_img_path,people_img_path,back_image_path,pose_img_path,local_img_dir=i.split(',')
            people_img_path=os.path.splitext(people_img_path)[0]
            people_img_path=people_img_path.split('/')[:-3]+['groundsam_people_img']+people_img_path.split('/')[-2:]
            people_img_path='/'.join(people_img_path)+'.png'
            
            if not ( os.path.isdir(local_img_dir) and os.path.isfile(people_img_path) and os.path.isfile(pose_img_path)
                    and os.path.isfile(back_image_path) and os.path.isfile(target_img_path)):
                print(people_img_path)
            else:
                c=0
                for i in os.listdir(local_img_dir):
                    c+=1
                if c<8:
                    print(local_img_dir)
                else:
                    self.data_pairs.append((target_img_path,people_img_path,back_image_path,pose_img_path,local_img_dir))
        
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
            raw,people,back,pose,local=self.data_pairs[index]
            back=Image.open(back)
            pose=Image.open(pose)
            raw=Image.open(raw)
            
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
            people_clip=self.transformer_clip(people).pixel_values[0]
            back=self.augmentation(back,transform1,self.transformer_ae,state)
            # ref_local_img=[]
            # for i in os.listdir(local):
            #     local_img=cv2.imread(os.path.join(local,i))
            #     local_img=cv2.cvtColor(local_img,cv2.COLOR_BGR2RGB)
            #     ref_local_img.append(self.augmentation(local_img,None,self.transformer_ae,state))
            # ref_local_img=torch.cat(ref_local_img,dim=0)
            
        except Exception as e:
            print(raw)
            traceback.print_exc()
        return back,people_vae,people_clip,pose,raw