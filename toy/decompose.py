import openql as ql
import numpy as np
import torch,shutil
import math,os,sys
from scipy.stats import unitary_group



def decompose(U,file_to_write):
    assert U.shape[0]==U.shape[1],            'not square input '
    assert math.log2(U.shape[0]).is_integer(), 'not the right shape'
    nqubits = int(math.log2(U.shape[0]))
    U=U.flatten()
    ql.initialize()
    ql.set_option('output_dir', 'output')
    ql.set_option('log_level', 'LOG_ERROR')
    platform = ql.Platform('my_platform', 'none')
    program = ql.Program('my_program', platform, nqubits)
    kernel = ql.Kernel('my_kernel', platform, nqubits)
    unitary = ql.Unitary('u1',U)
    unitary.decompose()
    kernel.gate(unitary, list(range(nqubits)))
    program.add_kernel(kernel)
    program.compile()
    with open('output/my_program.qasm') as f:
        lines = f.readlines()[7:]
        lines=[l[4:] for l in lines]
    with open(file_to_write, 'w') as f2:
        f2.write(''.join(lines))
    shutil.rmtree('output')

def batch_decompose(U_batch,folder):
    print(20*'-')
    print(folder)
    for i in range(U_batch.shape[0]):
        print(f'{i+1}/{U_batch.shape[0]}',end='\r')
        decompose(U_batch[i],folder+f'{i+1}.txt')
    print('done.')
    print(20*'-')
if __name__ == "__main__":
    for file in sys.argv[1:]:
        sH=torch.load(file)
        sH=sH.view(-1,sH.size(-1),sH.size(-1)).cdouble()
        U=torch.matrix_exp(sH-sH.permute(0,2,1).conj())

        folder=file.split('.')[0]+'/'
        if not os.path.isdir(folder): os.mkdir(folder)
        U=U.numpy()
        batch_decompose(U,folder)

