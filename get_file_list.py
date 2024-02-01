import os

src_dir='/data/zwplus/tiktok_image/test'

f=open('/data/zwplus/tiktok_image/test_list.txt','w')
for i in os.listdir(os.path.join(src_dir,'image')):
    image_dir=os.path.join(src_dir,'image',i)
    mask_dir=os.path.join(src_dir,'mask',i)
    dense_dir=os.path.join(src_dir,'dense',i)
    people_dir=os.path.join(src_dir,'people',i)
    back_dir=os.path.join(src_dir,'back',i)

    for j in os.listdir(image_dir):
        file_name=j.split('.')[0]

        image=os.path.join(image_dir,j)
        mask=os.path.join(mask_dir,file_name+'.png')
        dense=os.path.join(dense_dir,file_name+'.png')
        people=os.path.join(people_dir,file_name+'.png')
        back=os.path.join(back_dir,file_name+'.png')
        
        if os.path.isfile(mask) and os.path.isfile(dense) and os.path.isfile(back) :
            f.write(','.join([image,mask,dense,people,back])+'\n')
        else:
            print(image)

f.close()