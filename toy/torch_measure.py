from torch_ingredients import UnitaryN
import torch
import time

def measureTorchSpeed(NUMWIRES,NUMPOINTS,EPOCHS,bs=1,gpu=False):
    if gpu: device='cuda:0' 
    else: device= 'cpu'
    measurements=[]
    Circuit=UnitaryN(NUMWIRES).to(device)
    x=torch.rand(NUMPOINTS,NUMWIRES).to(device)
    opt=torch.optim.Adam(Circuit.parameters(),lr=1e-3)
    for epoch in range(EPOCHS):
        T0=time.time()
        for mini_batch in torch.split(x,bs):
            opt.zero_grad()
            L=((mini_batch-Circuit(mini_batch))**2).mean()
            #print('\r {} {}'.format(epoch,L),end='')
            L.backward()
            opt.step()
        measurements.append(time.time()-T0)
    return measurements