import torch
import wandb
import numpy as np
import random
from model import model
import sys,argparse


parser=argparse.ArgumentParser()

parser.add_argument('-dataset', type=str,choices={'mnist','deepsat'})
parser.add_argument('-Q', type=int,choices={0,1})
parser.add_argument('-seed', type=int)
parser.add_argument('-lr', type=str, default='1e-4')


args=parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)



DATASET=args.dataset


device='cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using', device)




run=wandb.init(project=f'QConv_{args.dataset}')
wandb.run.log_code(".")
wandb.config.update({"Q": bool(args.Q),'lr':args.lr})
m=model(Q=args.Q,dataset=args.dataset).to(device)
m.train(epochs=200,lr=float(args.lr))
run.finish()
