import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self,input_size=784,hidden_size=3,num_classes=10):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, num_classes)
                                 )
    def forward(self,x):
        return self.net(x)


