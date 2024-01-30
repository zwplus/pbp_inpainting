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
        self.data_dict_back={}
        self.data_dict_people={}
        self.if_train=if_train
        pairs_list=[]
        for i in data_pairs_txt_list:
            with open(i,'r') as f:
                pairs_list.extend(f.readlines())
        pairs_list.sort()
        
        for i in pairs_list:
            i=i.strip()
            target_img_path,people_img_path,back_image_path,pose_img_path=i.split(',')
            people_img_path=os.path.splitext(people_img_path)[0]
            people_img_path=people_img_path.split('/')[:-3]+['groundsam_people_img']+people_img_path.split('/')[-2:]
            people_img_path='/'.join(people_img_path)+'.png'

            file_dir=target_img_path.split('/')[-2]
            file_name=target_img_path.split('/')[-1]
            file_name=file_name.split('.')[0]+'.png'
            back_dir=back_image_path.split('/')[:-1]
            back_dir.append(file_name)
            target_back='/'.join(back_dir)
            people_dir=people_img_path.split('/')[:-1]
            people_dir.append(file_name)
            target_people='/'.join(people_dir)
            
            count=0
            if not ( os.path.isfile(people_img_path) and os.path.isfile(pose_img_path)
                    and os.path.isfile(back_image_path) and os.path.isfile(target_img_path) and os.path.isfile(target_people) and os.path.isfile(target_back) ):
                print(people_img_path)
                print(back_image_path)
                print(target_people)
                print(target_back)
                count+=1
            else:
                self.data_pairs.append((target_img_path,people_img_path,back_image_path,pose_img_path))
                if people_img_path not in self.data_dict_back.keys():
                    self.data_dict_back[people_img_path]=[]
                    self.data_dict_people[people_img_path]=[]
                self.data_dict_back[people_img_path].append(target_back)
                self.data_dict_people[people_img_path].append(target_people)
        print(count)
        for i in self.data_dict_back.keys():
            if len(self.data_dict_back[i])==0:
                print(i)
        
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
            raw,people,back,pose=self.data_pairs[index]
            pose=Image.open(pose)
            raw=Image.open(raw)
            if self.if_train==False:
                people=Image.open(people)
                back=Image.open(back)
                transform1=None
            else:
                back_index=random.randint(0,len(self.data_dict_back[people])-1)
                # print(len(self.data_dict_back[people]))
                back=self.data_dict_back[people][back_index]
                back=Image.open(back)
                people_index=random.randint(0,len(self.data_dict_people[people])-1)
                people=self.data_dict_people[people][people_index]
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
            
            
        except Exception as e:
            print(raw)
            traceback.print_exc()
        return back_vae,people_vae,people_clip,back_clip,pose,raw

# t=train_dataset=diffusion_dataset([
#     '/data/zwplus/tiktok/train/train_new_pose.txt',
# ])
# t=iter(t)
# for i in range(10000):
#     next(t)
