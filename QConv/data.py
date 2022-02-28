import torch,os
import torchvision
import pandas as pd
import shutil

def getData(dataset):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if dataset=='mnist':

        train_set = torchvision.datasets.MNIST(root='data/', train=True,  transform=torchvision.transforms.ToTensor(), download=True)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=60000, shuffle=True)

        test_set = torchvision.datasets.MNIST(root='data/', train=False,  transform=torchvision.transforms.ToTensor(), download=True)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=60000, shuffle=True)

        X_test,Y_test=next(iter(testloader))
        X_train,Y_train=next(iter(trainloader))

        X_test=X_test.to(device)
        X_train=X_train.to(device)

        Y_test=Y_test.to(device)
        Y_train=Y_train.to(device)

    elif dataset=='deepsat':
        if not os.path.isdir('data/deepsat'):
            os.system('kaggle datasets download -d crawford/deepsat-sat6')
            os.system('mv deepsat-sat6.zip data/')
            shutil.unpack_archive('data/deepsat-sat6.zip', 'data/deepsat')
        X_train = pd.read_csv('data/deepsat/X_train_sat6.csv', header = None).values.reshape([-1,28,28,4])
        Y_train = pd.read_csv('data/deepsat/y_train_sat6.csv', header = None).values

        

        X_test = pd.read_csv('data/deepsat/X_test_sat6.csv', header = None).values.reshape([-1,28,28,4])
        Y_test = pd.read_csv('data/deepsat/y_test_sat6.csv', header = None).values

        X_test=(torch.tensor(X_test)/255).to(device).permute(0,3,1,2)
        X_train=(torch.tensor(X_train)/255).to(device).permute(0,3,1,2)

        Y_train=torch.tensor(Y_train).max(1).indices
        Y_test=torch.tensor(Y_test).max(1).indices

        Y_test=Y_test.to(device)
        Y_train=Y_train.to(device)


    return X_test,Y_test,X_train,Y_train
