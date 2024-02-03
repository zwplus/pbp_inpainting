import torch
from collections import OrderedDict
ck=torch.load('/home/user/zwplus/pbp_inpainting/final.bin')
print(ck['controlnet_cond_embedding.conv_in.weight'].shape)
# print(ck['conv_in.weight'].shape)

# unet_dict=OrderedDict()
# controlnet_dict=OrderedDict()
# people_proj=OrderedDict()
# people_global_fusion=OrderedDict()
# people_local_fusion=OrderedDict()

# for i in ck.keys():
#     if 'controlnet_cond_embedding.conv_in.weight' in i:
#         conv_new_2 = torch.nn.Conv2d(
#                 in_channels=3,
#                 out_channels=16, 
#                 kernel_size=3,
#                 padding=1,
#             )

#         torch.nn.init.kaiming_normal_(conv_new_2.weight)
#         ck[i]=torch.cat([ck[i],conv_new_2.weight.data],dim=1)


# torch.save(ck,'/home/user/zwplus/pbp_inpainting/final.bin')
# torch.save(controlnet_dict,'/root/data1/github/pbp_inpainting/sd-2.1/test/control.bin')
# torch.save(people_proj,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_proj.bin')
# torch.save(people_local_fusion,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_local_fusion.bin')
# torch.save(people_global_fusion,'/root/data1/github/pbp_inpainting/sd-2.1/test/people_global_fusion.bin')