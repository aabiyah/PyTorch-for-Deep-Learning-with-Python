
```
import torch
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_cached()

a = torch.FloatTensor([1.0, 2.0])
a
a.device

a = torch.FloatTensor([1.0, 2.0]).cuda()
a.device
torch.cuda.memory_allocated()

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

class Model(nn.Module):
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__()
    self.fc1 = nn.Linear(in_features,h1)  # input layer
    self.fc2 = nn.Linear(h1,h2)           # hidden layer
    self.out = nn.Linear(h2,out_features) # output layer

  def forward(self, x):
    x = F.relu(self.fc1(x)
    x = F.relu(self.fc2(x)
    x = self.out(x)
    return x

torch.manual_seed(32)
model = Model()

next(model.parameters()).is_cuda
# False

gpumodel = model.cuda()

next(model.parameters()).is_cuda
# True

df = pd.read_csv(../Data/iris.csv)

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test,
```
