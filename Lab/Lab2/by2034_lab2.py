import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse
import time
#%matplotlib inline


parse = argparse.ArgumentParser(description = 'Pytorch HPC CIFAR Training')
parse.add_argument('--lr', default = 0.1, type = float, help = 'learning rate')
parse.add_argument('--opt', default = 'sgd', type = str, help = 'optimizer')
parse.add_argument('--num_workers', default = 0, type = int, help = 'num_workers')
parse.add_argument('--batch_size', default = 128, type = int, help = 'batch_size')

args = parse.parse_args()
lr = args.lr
opt = args.opt
num_workers = args.num_workers
batch_size = args.batch_size
print(lr, opt, num_workers, batch_size)
# Load dataset
'''
We will use CIFAR10, which contains 50K 32 x 32 color images
'''
trainsform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainsform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform=trainsform_train)
test_set = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform=trainsform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,shuffle = True, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#print(classes)


# Build model
'''
Create ResNet18. Specifically, The first convolutional layer should have 3 input channels, 64 output channels, 3x3 kernel, with stride=1 and padding=1. Followed by 8 basic blocks in 4 subgroups (i.e. 2 basic blocks in each subgroup):

1. The first sub-group contains a convolutional layer with 64 output channels, 3x3 kernel, stride=1, padding=1.
2. The second sub-group contains a convolutional layer with 128 output channels, 3x3 kernel, stride=2, padding=1.
3. The third sub-group contains a convolutional layer with 256 output channels, 3x3 kernel, stride=2, padding=1.
4. The fourth sub-group contains a convolutional layer with 512 output channels, 3x3 kernel, stride=2, padding=1.
5. The final linear layer is of 10 output classes. For all convolutional layers, use RELU activation functions, and use batch normal layers to avoid covariant shift. Since batch-norm layers regularize the training, set the bias to 0 for all the convolutional layers. Use SGD optimizers with 0.1 as the learning rate, momentum 0.9, weight decay 5e-4. The loss function is cross-entropy.

For all convolutional layers, use RELU activation functions, and use batch normal layers to avoid covariant shift. Since batch-norm layers regularize the training, set the bias to 0 for all the convolutional layers.
'''

## Basic block
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, input_channels, out_channels, stride = 1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    # when stride != 1 or input_channels != out_channels, it means the width and height are different
    if stride != 1 or input_channels != self.expansion * out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(input_channels, self.expansion * out_channels, kernel_size = 1, stride = stride, bias = False),
          nn.BatchNorm2d(self.expansion * out_channels)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

## ResNet
class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes = 10):
    super(ResNet, self).__init__()
    self.input_channels = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.input_channels, out_channels, stride))
      self.input_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

def ResNet18():
  return ResNet(BasicBlock, [2,2,2,2])

#net = ResNet18()
#print(net)

# C1 Train in Pytorch
'''
Create a main function that creates the DataLoaders for the training set and the neural network, then runs 5 epochs with a complete training phase on all the mini-batches of the training set. Write the code as device-agnostic, use the ArgumentParser to be able to read parameters from input, such as the use of Cuda, the data_path, the number of data loader workers, and the optimizer (as string, eg:'sgd')
For each minibatch calculate the training loss value, the top-1 training accuracy of the predictions, measured on training data.
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#net = net.to(device)
weight_decay = 5e-4
loss_fn = nn.CrossEntropyLoss()

if opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=weight_decay)
elif opt == 'sgd_nesterov':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=weight_decay, nesterov = True)
elif opt == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=lr,weight_decay=weight_decay)
elif opt == 'Adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=lr,weight_decay=weight_decay)
elif opt == 'Adam':
   optimizer = optim.Adam(net.parameters(), lr=lr, betas = (0.9, 0.99), weight_decay=weight_decay)

def train(epoch, train_loss_history, train_acc_history, data_loading_time, mini_training_time_total_epoch):
    print('\nEpoch: %d' % epoch)
    net_withoutBN.train()
    train_loss = 0
    correct = 0
    total = 0
    data_loading_time_total = 0
    mini_training_time = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_loading_time_start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        data_loading_time_end = time.time()
        data_loading_time_total += (data_loading_time_end - data_loading_time_start)

        mini_training_time_start = time.time()
        optimizer.zero_grad()
        outputs = net_withoutBN(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        mini_training_time_end = time.time()
        mini_training_time.append(mini_training_time_end - mini_training_time_start)

        train_loss += loss.item()
        train_loss_history.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc_history.append(100. * correct / total)
        '''
        print("\nThe batch index: {0:d}, len of train loader: {1:d}, Loss: {2:.3f}, acc: {3:.3f}".format(batch_idx,
                                                                                             len(train_loader),
                                                                                             train_loss / (batch_idx + 1),
                                                                                             100. * correct / total)
          )
        '''
    data_loading_time.append(data_loading_time_total)
    mini_training_time_total_epoch.append(mini_training_time)
    

test_loss_history = []
test_acc_history = []
train_loss_history = []
train_acc_history = []
total_train_time_epoch = []
data_loading_time = []
mini_training_time_total_epoch = []

epoch = 5
for epo in range(epoch):
  train_time_start = time.time()
  train(epo, train_loss_history, train_acc_history, data_loading_time, mini_training_time_total_epoch)
  train_time_end = time.time()
  total_train_time_epoch.append(train_time_end - train_time_start)

#print(train_loss_history)
for i in range(0, 1956, 391):
    print(np.mean(train_acc_history[i:i+391]))
    print(np.mean(train_loss_history[i:i+391]))
print(total_train_time_epoch)

# C3: I/O optimiation starting from code in C2

def num_worker_time(num_worker_data_loading_time):
  data_loading_time_total = 0
  data_loading_time_start = time.time()
  temp = 0
  for batch_idx, (inputs, targets) in enumerate(train_loader):  
    in_temp_start = time.time()
    inputs, targets = inputs.to(device), targets.to(device)   
    in_temp_end = time.time()
    temp += (in_temp_end - in_temp_start)
  data_loading_time_end = time.time()  
  whole_time = data_loading_time_end - data_loading_time_start
  num_worker_data_loading_time.append(whole_time - temp)

num_workers = [0,1,2,4,8,12,16]
batch_size = 128
num_worker_data_loading_time = []
for i in num_workers:
  print("new net: {}".format(i))
  train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,shuffle = True, num_workers = i)
  num_worker_time(num_worker_data_loading_time)
print(num_worker_data_loading_time)


# C7: Experimenting without Batch Norm layer
'''
class BasicBlock_withoutBN(nn.Module):
  expansion = 1

  def __init__(self, input_channels, out_channels, stride = 1):
    super(BasicBlock_withoutBN, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
    # self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
    # self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    # when stride != 1 or input_channels != out_channels, it means the width and height are different
    if stride != 1 or input_channels != self.expansion * out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(input_channels, self.expansion * out_channels, kernel_size = 1, stride = stride, bias = False),
          # nn.BatchNorm2d(self.expansion * out_channels)
      )

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.conv2(out)
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet_withoutBN(nn.Module):
  def __init__(self, block, num_blocks, num_classes = 10):
    super(ResNet_withoutBN, self).__init__()
    self.input_channels = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
    # self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.input_channels, out_channels, stride))
      self.input_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

def ResNet18_withoutBN():
  return ResNet_withoutBN(BasicBlock_withoutBN, [2,2,2,2])

net_withoutBN = ResNet18_withoutBN()

if opt == 'sgd':
    optimizer = optim.SGD(net_withoutBN.parameters(), lr=lr, momentum=0.9,weight_decay=weight_decay)
elif opt == 'sgd_nesterov':
    optimizer = optim.SGD(net_withoutBN.parameters(), lr=lr, momentum=0.9,weight_decay=weight_decay, nesterov = True)
elif opt == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=lr,weight_decay=weight_decay)
elif opt == 'Adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=lr,weight_decay=weight_decay)
elif opt == 'Adam':
   optimizer = optim.Adam(net.parameters(), lr=lr, betas = (0.9, 0.99), weight_decay=weight_decay)

epoch = 5
for epo in range(epoch):
  train_time_start = time.time()
  train(epo, train_loss_history, train_acc_history, data_loading_time, mini_training_time_total_epoch)
  train_time_end = time.time()
  total_train_time_epoch.append(train_time_end - train_time_start)
#print(train_loss_history)
for i in range(0, 1956, 391):
    print(np.mean(train_acc_history[i:i+391]))
    print(np.mean(train_loss_history[i:i+391]))
print(total_train_time_epoch)
'''

# Q3
from torchsummary import summary
summary(net, (3,32,32))

num_params = sum(param.numel() for param in net.parameters() if param.requires_grad)
print(num_params)
