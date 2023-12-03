import torch
from collections import OrderedDict
ck=torch.load('/home/user/zwplus/pbp_inpainting/sd-2.1/fp32/unet-inpainting/diffusion_pytorch_model.bin')
print(ck.keys())
print(ck['conv_in.weight'].shape)

# unet_dict=OrderedDict()
# controlnet_dict=OrderedDict()
# people_proj=OrderedDict()
# people_global_fusion=OrderedDict()
# people_local_fusion=OrderedDict()

# for i in ck.keys():
#     if 'unet' in i:
#         t=i.split('.')[1:]
#         n='.'.join(t)
#         print(n)
#         unet_dict[n]=ck[i]
#     elif 'controlnet' in i:
#         t=i.split('.')[1:]
#         n='.'.join(t)
#         controlnet_dict[n]=ck[i]
#     elif 'people_proj' in i:
#         t=i.split('.')[1:]
#         n='.'.join(t)
#         people_proj[n]=ck[i]
#     elif 'people_local_fusion' in i:
#         t=i.split('.')[1:]
#         n='.'.join(t)
#         people_local_fusion[n]=ck[i]
#     elif 'people_global_fusion' in i:
#         t=i.split('.')[1:]
#         n='.'.join(t)
#         people_global_fusion[n]=ck[i]

# torch.save(unet_dict,'/root/data1/github/pbp_inpainting/sd-2.1/test/unet.bin')
# torch.save(controlnet_dict,'/root/data1/github/pbp_inpainting/sd-2.1/test/control.bin')
# torch.save(people_proj,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_proj.bin')
# torch.save(people_local_fusion,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_local_fusion.bin')
# torch.save(people_global_fusion,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_global_fusion.bin')