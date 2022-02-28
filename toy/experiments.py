from torch_measure import measureTorchSpeed
from pennyLane_measure import measurePennyLaneSpeed
import matplotlib.pyplot as plt
import numpy as np


def plot(x,measurements,label,c):
    measurements=np.array(measurements)
    mean=measurements.mean(1)
    std=measurements.std(1) 
    plt.plot(x,mean,label=label,alpha=.5,c=c)
    plt.fill_between(x,mean-std, mean+std, alpha=0.2,color=c)
    plt.scatter(x,mean,marker='x',c=c)

wires=[1,2,3,4,5,6,7,8,9,10]
results=np.zeros((len(wires),8))

# warmup
PL=[measurePennyLaneSpeed(NUMWIRES=w,NUMPOINTS=1,EPOCHS=10) for w in wires[:3]]
T=[measureTorchSpeed(NUMWIRES=w,NUMPOINTS=1,EPOCHS=10) for w in wires]
#


PL=[measurePennyLaneSpeed(NUMWIRES=w,NUMPOINTS=1,EPOCHS=10) for w in wires[:8]]
T=[measureTorchSpeed(NUMWIRES=w,NUMPOINTS=1,EPOCHS=10) for w in wires]

results[:len(PL),0]=32*np.array(PL).mean(1)
results[:len(PL),1]=32*np.array(PL).std(1)

results[:,2]=32*np.array(T).mean(1)
results[:,3]=32*np.array(T).std(1)


plot(wires[:8],PL,'PennyLane',c='C0')
plot(wires,T,'PyTorch',c='C1')
plt.xlabel('Number of Wires')
plt.xticks(wires)
plt.ylabel('Time (s)')
plt.yscale('log')
plt.ylim(1e-4,1e2)
plt.legend(loc='best')
plt.savefig('PennyLane_PyTorch.pdf')
plt.close()
print('Exp1 done.')





T=[measureTorchSpeed(NUMWIRES=w,NUMPOINTS=32,EPOCHS=10) for w in wires]
T_batch=[measureTorchSpeed(NUMWIRES=w,NUMPOINTS=32,EPOCHS=10,bs=32) for w in wires]
T_batch_GPU=[measureTorchSpeed(NUMWIRES=w,NUMPOINTS=32,EPOCHS=10,bs=32,gpu=True) for w in wires]

results[:,4]=np.array(T_batch).mean(1)
results[:,5]=np.array(T_batch).std(1)

results[:,6]=np.array(T_batch_GPU).mean(1)
results[:,7]=np.array(T_batch_GPU).std(1)



plot(wires,T,'PyTorch (batch size=1, CPU)',c='C1')
plot(wires,T_batch,'PyTorch (batch size=32, CPU)',c='C2')
plot(wires,T_batch_GPU,'PyTorch (batch size=32, GPU)',c='C3')
plt.xlabel('Number of Wires')
plt.xticks(wires)
plt.ylabel('Time (s)')
plt.yscale('log')
plt.ylim(1e-4,1e3)
plt.legend(loc='best')
plt.savefig('PyTorch_variants.pdf')

print('Exp2 done.')

np.save('results', results)