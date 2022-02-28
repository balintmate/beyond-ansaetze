import numpy as np
results=np.load('results.npy')
results[8:,0]=float('inf')


for numwires in range(results.shape[0]):
    print(f'{numwires+1} & ', end='')
    for i in range(4):
        if results[numwires][::2].argmin()==i:
             print(' \\textbf{',end='')
             print(f'{results[numwires][2*i]:.2e}',end='')
             print('}',end='')
             print(f'$\pm$ {results[numwires][2*i+1]:.2e} ',end='')
        else:
            print(f'{results[numwires][2*i]:.2e} $\pm$ {results[numwires][2*i+1]:.2e} ',end='')
        if i<3: print('&',end='') 
    print('\\\\')