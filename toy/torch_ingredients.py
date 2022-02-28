import torch
import numpy as np




class UnitaryN(torch.nn.Module):
    def __init__(self,wires):
        super().__init__()
        D=2**wires
        self.params = torch.nn.Parameter(torch.randn(D,D))
    def forward(self,x):
        x=encode(x)
        sym=self.params+self.params.T
        asym=self.params-self.params.T
        SH = torch.view_as_complex(torch.cat((asym.unsqueeze(-1),sym.unsqueeze(-1)),-1))
        U=torch.matrix_exp(SH)
        x=torch.einsum('ij,bi->bj',(U,x))
        x=decode(x)
        return x

def RX(theta):
    c=(theta/2).cos().unsqueeze(-1).unsqueeze(-1)
    s=(theta/2).sin().unsqueeze(-1).unsqueeze(-1)
    i=torch.view_as_complex(torch.tensor([0., 1.]))
    row1=torch.cat((c,-i*s),-1)
    row2=torch.cat((-i*s,c),-1)
    return torch.cat((row1,row2),-2)

def tensor_product(A,B): 
     res = torch.einsum('bac,bkp->bakcp', A, B)
     res = res.view(A.size(0), A.size(1)*B.size(1),A.size(2)*B.size(2)) 
     return res 

def encode(x):
    R_per_wire=RX(x)

    U=R_per_wire[:,0]
    for i in range(R_per_wire.size(1)-1):
        U=tensor_product(U,R_per_wire[:,i+1])
    
    ground_state=torch.zeros(2**x.size(1), dtype=torch.cfloat).to(x.device)
    ground_state[0]=1
    encoded_data=torch.einsum('bij,i->bj',(U,ground_state))
    return encoded_data

def decode(x):
    N=int(np.log2(x.size(1)))
    probs=(x*x.conj()).abs()
    sigma_x=torch.tensor([1,-1]).to(x.device)
    probs=probs.view((probs.size(0),)+N*(2,))
    #print((probs*sigma_x).sum().acos())
    out=[]
    for j in range(N):  
        indices_to_sum=tuple([i+1 for i in range(j)]+[i+1 for i in range(j+1,N)])
        if indices_to_sum != ():
            out.append((probs.sum(indices_to_sum)*sigma_x).sum(1,keepdim=True).acos())
        else:
            out.append((probs*sigma_x).sum(1,keepdim=True).acos())
    out=torch.cat(out,1)
    return out


# data=torch.rand(128,N)*np.pi 
# ((data-decode(encode(data)))**2).sum()



# N=2                              
# D=2**N  
# x = torch.randn(D,D, dtype=torch.cfloat) 
# SH = x - x.T.conj()
# U=torch.matrix_exp(SH)
# #this should print something small
# print((U@U.T.conj()-torch.eye(D)).abs().max())