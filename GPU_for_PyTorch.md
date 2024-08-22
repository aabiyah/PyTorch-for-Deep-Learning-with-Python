
This code demonstrates how to build, train, and evaluate a simple neural network using PyTorch with GPU acceleration. Below is a step-by-step explanation of the code:

### 1. Check CUDA Availability
```
import torch
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_cached()
```
* torch.cuda.is_available(): Checks if a CUDA-capable GPU is available.
* torch.cuda.current_device(): Returns the index of the current GPU.
* torch.cuda.get_device_name(): Retrieves the name of the GPU.
* torch.cuda.memory_allocated(): Checks the memory currently allocated on the GPU.
* torch.cuda.memory_cached(): Checks the memory cached on the GPU.

### 2. Create and Transfer Tensors to GPU
```
a = torch.FloatTensor([1.0, 2.0])
a
a.device

a = torch.FloatTensor([1.0, 2.0]).cuda()
a.device
torch.cuda.memory_allocated()
```
* Creates a 1D tensor a with elements [1.0, 2.0].
* a.device: Checks the device on which a is stored (CPU initially).
* Transfers a to the GPU using .cuda().
* a.device: Confirms that a is now on the GPU.
* Checks the memory allocated on the GPU after transferring a.

### 3. Import Required Libraries
```
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```
* Imports necessary PyTorch modules for building neural networks (nn and F), managing datasets (Dataset, DataLoader), and splitting data (train_test_split).
* Also imports pandas for data handling and matplotlib for visualization.

### 4. Define the Neural Network Model
```
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
```
* Defines a simple feedforward neural network with:
  1. fc1: First fully connected layer with in_features inputs and h1 outputs.
  2. fc2: Second fully connected layer with h1 inputs and h2 outputs.
  3. out: Output layer with h2 inputs and out_features outputs.
* The forward method defines the forward pass, applying the ReLU activation function to the first two layers and leaving the output layer linear.

### 5. Instantiate and Transfer the Model to GPU
```
torch.manual_seed(32)
model = Model()

next(model.parameters()).is_cuda
# False

gpumodel = model.cuda()

next(model.parameters()).is_cuda
# True
```
* torch.manual_seed(32): Sets a seed for reproducibility.
* Instantiates the Model and checks if its parameters are on the GPU (initially False).
* Transfers the model to the GPU with .cuda() and confirms the parameters are now on the GPU (True).

### 6. Load and Prepare the Dataset
```
df = pd.read_csv(../Data/iris.csv)

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 33

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()

trainloader = DataLoader(X_train, batch_size=60, shuffle=True, pin_memory=True)
testloader = DataLoader(X_test, batch_size=60, shuffle=False, pin_memory=True)
```
* Loads the Iris dataset from a CSV file.
* Splits the dataset into training and testing sets with train_test_split.
* Converts the features (X_train, X_test) and labels (y_train, y_test) into PyTorch tensors and transfers them to the GPU.
* Initializes DataLoader objects to manage batches for training and testing.

### 7. Define the Loss Function and Optimizer
```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
* criterion: Uses cross-entropy loss, suitable for classification
* tasks.optimizer: Uses the Adam optimizer to update the model's parameters, with a learning rate of 0.01.
  
### 8. Train the Model
```
import time
start = time.time()
epochs = 100
losses = []

for i in range(epochs):

  y_pred = gpumodel.forward(X_train)
  loss = criterion(y_pred, y_train)
  losses.append(loss)

  if i%10==0:
      print(f'Epoch: {i} loss {loss:item()}')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

end = time.time()-start
print(f'TOTAL TIME: {end}')
```
* Starts a timer to measure the total training time.
*  Runs a loop for 100 epochs to train the model:
  1. y_pred: Computes the model’s predictions on the training data.
  2. loss: Calculates the loss between predictions and actual labels.
  3. Stores the loss and prints it every 10 epochs.
  4. optimizer.zero_grad(): Clears gradients from the previous step.
  5. loss.backward(): Computes the gradient of the loss with respect to model parameters.
  6. optimizer.step(): Updates the model parameters based on the gradients.
* Stops the timer and prints the total training time.

### 9. Evaluate the Model
```
correct = 0
with torch.no_grad():
for i, data in enumerate(X_test):
  y_val = gpumodel.forward(data)
  print(f'{i+1:2}. {str(y_val):38 {y_test[i]}')
  if y_val.argmax().item() == y_test[i]:
      correct += 1
print(f'\n{correct} out of {len(y_test)} = {100*correct/len(y_test):.2f}% correct')
```
* Disables gradient calculation with torch.no_grad() to save memory and computation during evaluation.
* Iterates through the test set, making predictions with the trained model.
* Prints the predicted and actual labels for each test sample.
* Counts the number of correct predictions and calculates the accuracy as a percentage.

## Sumamry
This code builds and trains a simple neural network on the Iris dataset using PyTorch with GPU acceleration. It demonstrates key concepts like transferring data and models to the GPU, defining a neural network with multiple layers, and using a DataLoader for batching during training. Finally, it evaluates the model on test data and reports the accuracy.
