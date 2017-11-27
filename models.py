import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dp1 = nn.Dropout2d(p=0.1)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True)
        self.fc0 = nn.Linear(4*4*50,100)
        self.fc1 = nn.Linear(100,27)


    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2,2)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2,2)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc0(x.view(x.size(0), -1))
        x = self.fc1(x.view(x.size(0), -1))

        return F.log_softmax(x)

