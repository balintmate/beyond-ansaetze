import torch
import wandb
from math import prod
import torch.nn as nn
import numpy as np
from einops import rearrange

class QConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.cuda_device='cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        numwires=prod(self.kernel_size)
        if out_channels%(numwires)!=0:
            print('\033[91m'+'number of qubits dont divide the number of output channels!')
        else:
            #classical conv will have in_channels*out_channels*prod(kernel_size) parameters
            self.params = torch.nn.Parameter(torch.randn(in_channels,out_channels//numwires,2**numwires,2**numwires))
            print(f'Qconv layer with {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters')


    def forward(self,x):
        
        #b c x y
        patches=torch.cat([x[:,:,i:i+self.kernel_size[0]].unsqueeze(-1) for i in range(0,x.size(2)+1-self.kernel_size[0],self.stride[0])],-1)
        patches=torch.cat([patches[:,:,:,i:i+self.kernel_size[1]].unsqueeze(-1) for i in range(0,x.size(3)+1-self.kernel_size[1],self.stride[1])],-1)
        #b c x y px py
        patches_size=patches.size()
        patches=patches.permute(0,1,4,5,2,3).flatten(4)
        patches=rearrange(patches,'b c px py i-> c (b px py) i')
        x=encode(patches)

        sym=self.params+self.params.permute(0,1,3,2)
        asym=self.params-self.params.permute(0,1,3,2)
        SH = torch.view_as_complex(torch.cat((asym.unsqueeze(-1),sym.unsqueeze(-1)),-1))
       
        U=torch.matrix_exp(SH)
        x=torch.einsum('cdij,cbi->cdbj',(U,x))
        x=decode(x)
        x=x.sum(0) #weighted sum?
        convolved_patches=rearrange(x,'c (b px py) (x y) -> b (c x y) px py',b=patches_size[0],px=patches_size[4],x=patches_size[2])
        return convolved_patches


def RX(theta):
    c=(theta/2).cos().unsqueeze(-1).unsqueeze(-1)
    s=(theta/2).sin().unsqueeze(-1).unsqueeze(-1)
    i=torch.view_as_complex(torch.tensor([0., 1.]))
    row1=torch.cat((c,-i*s),-1)
    row2=torch.cat((-i*s,c),-1)
    return torch.cat((row1,row2),-2)

def tensor_product(A,B): 
    res = torch.einsum('ijac,ijkp->ijakcp', A, B)
    res = res.view(res.size(0),res.size(1), A.size(2)*B.size(2),A.size(3)*B.size(3)) 
    return res 

def encode(x):
    R_per_wire=RX(x)
    U=R_per_wire[:,:,0]
    for i in range(R_per_wire.size(2)-1):
        U=tensor_product(U,R_per_wire[:,:,i+1])
    ground_state=torch.zeros(2**x.size(2), dtype=torch.cfloat).to(x.device)
    ground_state[0]=1
    encoded_data=torch.einsum('cbij,i->cbj',(U,ground_state))
    return encoded_data

def decode(x):
    #c d b i
    N=int(np.log2(x.size(-1)))
    probs=(x*x.conj()).abs()
    sigma_x=torch.tensor([1,-1]).to(x.device)
    probs=probs.view(probs.size()[:-1]+N*(2,))
    #print((probs*sigma_x).sum().acos())
    out=[]
    for j in range(N):  
        indices_to_sum=tuple([i+3 for i in range(j)]+[i+3 for i in range(j+1,N)])
        if indices_to_sum != ():
            out.append((probs.sum(indices_to_sum)*sigma_x).sum(-1,keepdim=True).acos())
        else:
            out.append((probs*sigma_x).sum(-1,keepdim=True).acos())
    out=torch.cat(out,-1)
    return out

