import os
import cv2
import numpy as np
from tqdm import tqdm

dir_list=[]
for i in os.listdir('/data/zwplus/tiktok_image/train/image'):
    dir_list.append(i)


for i in tqdm(dir_list):
    mask_dir=os.path.join("/data/zwplus/tiktok_image/train/mask",i)
    img_dir=os.path.join("/data/zwplus/tiktok_image/train/image",i)

    back_dir=os.path.join("/data/zwplus/tiktok_image/train","back",i)
    app_dir=os.path.join("/data/zwplus/tiktok_image/train","people",i)
    os.makedirs(back_dir,exist_ok=True)
    os.makedirs(app_dir,exist_ok=True)

    for j in os.listdir(img_dir):
        try:
            file_name=j.split('.')[0]
            if not os.path.isfile(os.path.join(mask_dir,file_name+'.png')) or os.path.isfile(os.path.join(app_dir,file_name+'.png')):
                print(os.path.join(mask_dir,file_name+'.png'))
                continue

            img=cv2.imread(os.path.join(img_dir,j))
            mask=cv2.imread(os.path.join(mask_dir,file_name+'.png'))

            mask[mask==255]=1
            mask[mask==0]=0


            people=(img*mask).astype(np.uint8)

            mask=1-mask
            back=(img*mask).astype(np.uint8)

            cv2.imwrite(os.path.join(app_dir,file_name+'.png'),people)
            cv2.imwrite(os.path.join(back_dir,file_name+'.png'),back)
        except Exception as e:
            print(e)
            print(os.path.join(mask_dir,file_name+'.png'))

