import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple 
import os
import requests
from tqdm import tqdm
from utils.discriminator import Discriminator

class NetLinLayer(nn.Module):
    def __init__(self,in_channels,out_channels=1) -> None:
        super().__init__()
        self.model=nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels,out_channels,1,1,0,bias=False)
        )

class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('shift',torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale',torch.Tensor([-.458,-.448,-.450])[None,:,None,None])

    def forward(self,x):
        return (x-self.shift)/self.scale

#通道维度归一化
def normalize_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    #计算感知损失时需要在通道维度上归一化
    norm_factor=torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+1e-10)

#空间维度求平均
def spatial_average(x):
    """
    imgs have: batch_size x channels x width x height --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2,3],keepdim=True)

class vgg16(torch.nn.Module):
    def __init__(self,requires_grad=False,pretrained=True) -> None:
        super().__init__()

        vgg_pretrained_features=models.vgg16(pretrained=pretrained).features
        slices=[vgg_pretrained_features[i] for i in range(30)]
        self.slice1=nn.Sequential(*slices[0:4])
        self.slice2=nn.Sequential(*slices[4:9])
        self.slice3=nn.Sequential(*slices[9:16])
        self.slice4=nn.Sequential(*slices[16:23])
        self.slice5=nn.Sequential(*slices[23:30])

        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad=False
    
    def forward(self,x):
        h=self.slice1(x)
        h_relu1_2=h
        h=self.slice2(h)
        h_relu2_2=h
        h=self.slice3(h)
        h_relu3_3=h
        h=self.slice4(h)
        h_relu4_3=h
        h=self.slice5(h)
        h_relu5_3=h

        vgg_outputs=namedtuple('VggOutputs',['relu1_2','relu2_2','relu3_3','relu4_3','relu5_3'])
        out=vgg_outputs(h_relu1_2,h_relu2_2,h_relu3_3,h_relu4_3,h_relu5_3)
        return out


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def get_ckpt_path(name,root):
    assert name in URL_MAP
    path=os.path.join(root,CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name],path)
    return path

def adopt_weight(disc_factor,i,threshold,value=0):
        if i < threshold:
            disc_factor=value
        return disc_factor

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class LPIPS(nn.Module):
    def __init__(self,use_dropout=True) -> None:
        super().__init__()
        self.scaling_layer=ScalingLayer()
        self.chns=[64,128,256,512,512]
        self.net=vgg16(pretrained=True,requires_grad=False)

        #lin实际上是求各个通道的权重
        self.lin0=NetLinLayer(self.chns[0])
        self.lin1=NetLinLayer(self.chns[1])
        self.lin2=NetLinLayer(self.chns[2])
        self.lin3=NetLinLayer(self.chns[3])
        self.lin4=NetLinLayer(self.chns[4])
        self.lins=[self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]

        self.load_from_pretrained()

        for parameters in self.parameters():
            parameters.requires_grad=False
    
    def load_from_pretrained(self,name="vgg_lpips"):
        ckpt=get_ckpt_path(name,'vgg_lpips')
        self.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')),strict=False)

    def forward(self,real_x,fake_x):

        #首先利用vgg16求得两个图像在不同尺度上的特征差
        #要沿着通道维归一化
        feature_real=self.net(self.scaling_layer(real_x))
        feature_fake=self.net(self.scaling_layer(fake_x))
        
        diffs={}
        for i in range(len(self.chns)):
            diffs[i]=(normalize_tensor(feature_real[i])-normalize_tensor(feature_fake[i]))**2
        
        #因为特征差的是多通道的，所以利用1*1的卷积求得每个通到的加权和，然后再在空间维度上求平均
        #z最后求和
        return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.chns))])
    
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                disc_loss="hinge",training=True):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight            #KL散度的约束   
        self.pixel_weight = pixelloss_weight  #L1LOSS权重
        self.perceptual_loss = LPIPS().eval()   
        self.perceptual_weight = perceptual_weight  #LPIPS损失的权重
        self.training=training
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = Discriminator(input_nc=disc_in_channels,
                                                n_layers=disc_num_layers,
                                                ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # print('rec_loss',torch.isinf(rec_loss).any())
        # print('rec_loss',torch.isnan(rec_loss).any())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # print('p_loss',torch.isinf(p_loss).any())
            # print('p_loss',torch.isnan(p_loss).any())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        # print('rec_loss',torch.isinf(rec_loss).any())
        # print('rec_loss',torch.isnan(rec_loss).any())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # print('nll_loss',torch.isinf(nll_loss).any())
        # print('nll_loss',torch.isnan(nll_loss).any())
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # print('kl_loss',torch.isinf(kl_loss).any())
        # print('kl_loss',torch.isnan(kl_loss).any())
        # print('weighted_nll_loss',torch.isinf(weighted_nll_loss).any())
        # print('weighted_nll_loss',torch.isnan(weighted_nll_loss).any())

        if optimizer_idx ==0:
                logits_fake = self.discriminator(reconstructions.contiguous())

                g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
                loss_dict={
                    '{}/nll_loss'.format(split):nll_loss.detach().mean(),
                    '{}/kl_loss'.format(split):kl_loss.detach().mean(),
                    '{}/g_loss'.format(split):g_loss.detach()
                }
                return loss,loss_dict
        elif optimizer_idx==1:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
                loss_dict={
                    '{}/d_loss'.format(split):d_loss.clone().detach()
                }
                return d_loss,loss_dict