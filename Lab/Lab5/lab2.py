# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:17:43 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import argparse
import time
from resnet import*
#from nobatchnorm import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--device', default='cuda', help='computing device')
parser.add_argument('--EPOCH', default=100, help='computing device')
parser.add_argument('--workers', default=2, help='computing device')
parser.add_argument('--optimizer', default='SGD', help='computing device')
parser.add_argument('--order', default=0, help='computing device')
args = parser.parse_args()
EPOCH = args.EPOCH
device = args.device if torch.cuda.is_available() else 'cpu'
workers = int(args.workers)
#print("device type:",device)
#print("Number of workers:", workers)
#print("optimizer", args.optimizer)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=workers)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=workers)



classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

#elif args.optimizer == 'Nesterov':
#    optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                          momentum=0.9, weight_decay=5e-4,nesterov=True)
#elif args.optimizer == 'Adagrad':
#    optimizer = optim.Adagrad(net.parameters(), lr=args.lr,
#                            weight_decay=5e-4 )
#elif args.optimizer == 'Adadelta':
#    optimizer = optim.Adadelta(net.parameters(), lr=args.lr,
#                            weight_decay=5e-4)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if __name__ == "__main__":
    acc = 0
    testacc = 0
    totaltime = 0
    
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    
    ts = time.time()
    for epoch in range(EPOCH):
        
        print('\nEpoch: %d' % (epoch + 1))
        epochtime = 0
        epochstart=time.perf_counter()
        
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        traintime = 0.0
        loadtime = 0.0
        
        
        loadstart=time.perf_counter()
        for i, data in enumerate(trainloader, 0):
            
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loadend=time.perf_counter()
            
            
            optimizer.zero_grad()
            trainstart=time.perf_counter()


            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            trainend=time.perf_counter()
            
            sum_loss += loss.item()
            traintime += trainend-trainstart
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            loadtime += loadend - loadstart
            #print(trainend - trainstart)
            loadstart=time.perf_counter()
        

        epochend=time.perf_counter()
        epochtime+=epochend-epochstart
        acc = max(acc,correct/total)
        totaltime+=epochtime
        print("loss: ",sum_loss/total)
        print("accuracy: ",acc)
        #print("epochtime", epochtime)
        #print("traintime:", traintime)
        #print("dataloadtime:", loadtime)

        



        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
            val_acc = correct/total
            print('val_acc',val_acc)
            if val_acc>=0.91:
                te = time.time()
                print('total time: ',te-ts)
                break
    PATH = str(args.order)+".pth"
    torch.save(net, PATH) 
#    print("average running time:",totaltime/EPOCH)
    print("best training accuracy:",acc)
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    