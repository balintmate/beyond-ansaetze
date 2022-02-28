import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import torch,time

def measurePennyLaneSpeed(NUMWIRES,NUMPOINTS,EPOCHS):
    measurements=[]
   
    dev = qml.device("default.qubit", wires=NUMWIRES)
    # match the dimensionality of the unitary group?
    params=torch.randn(2**NUMWIRES,2**NUMWIRES).requires_grad_()
    @qml.qnode(dev,interface="torch")

    def circuit(inputs,params):
        for k in range(NUMWIRES):
            qml.RX(np.pi * inputs[k], wires=k)
        RandomLayers(weights=params, wires=list(range(NUMWIRES)))
        return [qml.expval(qml.PauliZ(i)) for i in range(NUMWIRES)]

    opt = torch.optim.Adam([params], lr=1e-2)
    x=torch.rand(NUMPOINTS,NUMWIRES)
    for epoch in range(EPOCHS):
        T0=time.time()
        for mini_batch in torch.split(x,32):
            L=0
            for i in range(mini_batch.shape[0]):
                L=L+((circuit(mini_batch[i],params)-mini_batch[i])**2).mean()
            L.backward()
            opt.step()
            #print('\r {} {}'.format(epoch,L),end='')
        measurements.append(time.time()-T0)
    return measurements