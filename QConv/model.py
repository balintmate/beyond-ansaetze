import torch
import wandb
import torch.nn as nn
from data import getData
from Qconv import QConv2d
import sys
import os

class model(nn.Module):
    def __init__(self,Q,dataset):
        super().__init__()
        self.X_train,self.Y_train,self.X_test,self.Y_test=getData(dataset=dataset)
        if   dataset=='mnist':   numclasses=10
        elif dataset=='deepsat': numclasses=6

        if Q: self.N = nn.Sequential(
            nn.Conv2d(self.X_train.size(1),16,kernel_size=4,stride=2),nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,stride=1),nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=(2,2),stride=(2,2)),nn.ReLU(),
            QConv2d(16,8,kernel_size=(2,2),stride=(1,1)),
            nn.Flatten(1),
            nn.Linear(128,32),nn.ReLU(),
            nn.Linear(32,numclasses),nn.Softmax(1)
        )
            
        else: self.N=nn.Sequential(
            nn.Conv2d(self.X_train.size(1),32,kernel_size=4,stride=2),nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,stride=1),nn.ReLU(),
            nn.Conv2d(32,16,kernel_size=(2,2),stride=(2,2)),nn.ReLU(),
            nn.Conv2d(16,8,kernel_size=(2,2),stride=(1,1)),nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(128,32),nn.ReLU(),
            nn.Linear(32,numclasses),nn.Softmax(1)
            )

        self.Q=Q
        if Q and not os.path.isdir('tensors'): os.mkdir('tensors')
        wandb.config.update({"Num_params:": sum(p.numel() for p in self.parameters() if p.requires_grad)})

    def test(self,X,Y,s):
        score=0
        for mini_X,mini_Y in zip(torch.split(X,32),torch.split(Y,32)):
            labels=self.forward(mini_X)
            labels=torch.max(labels,dim=-1).indices
            score+=(labels==mini_Y).float().sum()
        wandb.log({s+'_acc':score/X.size(0)})

    def forward(self,x):
        x=self.N(x)
        return x

    def train(self,epochs,lr):
        optim=torch.optim.Adam(self.parameters(),lr=lr)
        for epoch in range(epochs):
            print(epoch+1,end=' ')
            sys.stdout.flush()
            for mini_X,mini_Y in zip(torch.split(self.X_train,32),torch.split(self.Y_train,32)):
                optim.zero_grad()
                L=nn.CrossEntropyLoss()(self.forward(mini_X),mini_Y)
                L.backward()
                wandb.log({'L':L.detach().cpu().numpy()})
                wandb.log({'epoch':epoch})
                optim.step()
            self.test(self.X_train,self.Y_train,'train')
            self.test(self.X_test,self.Y_test,'test')
    #         self.logQ(epoch)
    
    # def logQ(self,epoch):
    #     if not self.Q: return
    #     torch.save(self.N2[2].params.detach().cpu(),f'tensors/Q_{epoch}.tensor')
