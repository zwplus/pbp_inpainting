import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange,repeat
def nonlinearity(x):
    return x*torch.sigmoid(x)

def Normalize(in_channels,num_groups=32):
    return torch.nn.GroupNorm(num_groups,in_channels,eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self,in_channels,with_conv) -> None:
        super().__init__()
        self.with_conv=with_conv
        if self.with_conv:
            self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
    
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2.0,mode='nearest')
        if self.with_conv:
            x=self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self,in_channels,with_conv) -> None:
        super().__init__()
        self.with_conv=with_conv
        if self.with_conv:
            self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2)
    
    def forward(self,x):
        if self.with_conv:
            pad=(0,1,0,1)
            x=F.pad(x,pad,mode='constant',value=0)
            x=self.conv(x)
        else:
            x=F.avg_pool2d(x,kernel_size=2,stride=2)
        return x


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

#添加AdaIN模块，利用该模块来引入风格信息

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
