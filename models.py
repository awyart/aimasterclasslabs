import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(28*28,200)
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(100,27)


    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x.view(x.size(0), -1))

        return F.log_softmax(x)

