import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,input_nc,num_filters_last=64,n_layers=3) -> None:
        super().__init__()

        layers=[nn.Conv2d(input_nc,num_filters_last,4,2,1),nn.LeakyReLU(0.2)]
        num_filters_mult=1

        for i in range(1,n_layers+1):
            num_filters_mult_last=num_filters_mult
            num_filters_mult=min(2**i,4)

            layers+=[
                nn.Conv2d(num_filters_mult_last*num_filters_last,num_filters_last*num_filters_mult,4,2 if i<n_layers else 1,1,bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_filters_last*num_filters_mult)
            ]
        
        layers.append(nn.Conv2d(num_filters_last*num_filters_mult,1,4,1,1))
        self.model=nn.Sequential(*layers)
    
    def forward(self,x):
        return self.model(x)