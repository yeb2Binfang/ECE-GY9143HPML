'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
from time import perf_counter
from model import ResNet18
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--worker','-w', default=2, help='number of workers in dataloader')
parser.add_argument('--optimizer','-o', default="SGD", help='number of workers in dataloader')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device is %s"%(device))
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),#Random cropping, with size 32x32 and padding 4
    transforms.RandomHorizontalFlip(),#Random horizontal flipping with a probability of 0.5
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#RGB channel with mean and variance
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

try:
    num_workers = int(args.worker)
except:
    num_workers = 2

name_optimizer = args.optimizer
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=num_workers)#a minibatch size of 128 and 3 IO processes (i.e., num_workers=2).

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=num_workers)#a minibatch size of 100 and 3 IO processes (i.e., num_workers =2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

if name_optimizer == "Nesterov":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4,nesterov=True)
elif name_optimizer == "Adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr,
                      weight_decay=5e-4)
elif name_optimizer == "Adadelta":
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr,
                              weight_decay=5e-4)
elif name_optimizer == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                              weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    name_optimizer = "SGD"
print("Optimizer is %s, The number of workers is %d"%(name_optimizer, num_workers))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    training_time, batch_time = 0.0,0.0
    total_start_time = perf_counter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_training_time = perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        end_training_time = perf_counter()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        end_batch_time = perf_counter()

        training_time += end_training_time - start_training_time
        batch_time += end_batch_time-end_training_time
        # cur_training_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # if batch_idx == len(trainloader)-1:
        #     training_time = cur_training_time
    total_end_time = perf_counter()
    total_time = total_end_time-total_start_time-batch_time
    print("<tr><td>%d</td><td>%.6f s</td><td>%.6f s</td><td>%.6f s</td></tr>" % (epoch,total_time-training_time,training_time,total_time))
    return total_time, training_time, train_loss/len(trainloader), 100.*correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)

training_time = 0.0
total_time = 0.0
num_of_epoch = 5
train_loss = 0.0
accuracy = 0.0
for epoch in range(start_epoch, start_epoch+num_of_epoch):
    cur_total_time, cur_training_time,cur_loss,cur_accuracy = train(epoch)
    total_time += cur_total_time
    training_time += cur_training_time
    train_loss += cur_loss
    accuracy += cur_accuracy
    test(epoch)
    scheduler.step()
print("5 epoches: Training time: %.6f s, Data-loading time: %.6f s, Total Running time %.6f s, Average Running time %.6f s" % (training_time,total_time-training_time,total_time,total_time/num_of_epoch))
print(" Average Training time %.6f s, training loss: %.6f, top-1 training accuracy: %.6f%%" % (training_time/num_of_epoch,train_loss/num_of_epoch,accuracy/num_of_epoch))
print("<tr><td>%s</td><td>%.6f s</td><td>%.6f</td><td>%.6f %%</td></tr>" % (name_optimizer, training_time/num_of_epoch,train_loss/num_of_epoch,accuracy/num_of_epoch))
print("<tr><td>%d</td><td>%.6f s</td><td>%.6f s</td><td>%.6f s</td></tr>"%(num_workers,total_time,training_time,total_time-training_time))
print("The number of trainable parameters is", count_parameters(optimizer.param_groups[0]['params']))
print("The number gradients is", count_parameters(optimizer.param_groups[0]['params']))