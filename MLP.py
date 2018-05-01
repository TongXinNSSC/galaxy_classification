#coding:utf-8
from __future__ import print_function

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-path', default='checkpoint', type=str)
parser.add_argument('--load-path', default='', type=str)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
"""加载数据。组合数据集和采样器，提供数据上的单或多进程迭代器
参数：
dataset：Dataset类型，从其中加载数据
batch_size：int，可选。每个batch加载多少样本
shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
sampler：Sampler，可选。从数据集中采样样本的方法。
num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
collate_fn：callable，可选。
pin_memory：bool，可选
drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
"""
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.num_classes = num_classes

        self.stage1 = nn.Sequential(
            nn.Linear(784, 1000, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),)
        self.stage2 = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),)
        self.stage3 = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),)
        self.stage4 = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),)
        self.stage5 = nn.Sequential(
            nn.Linear(1000, num_classes, bias=True),
            nn.ReLU(inplace=True),

        )
        self.classifier = nn.Softmax(1)

    def forward(self, x):
        x = self.stage1(x)
        layer1_out = x
        x = self.stage2(x)
        layer2_out = x
        x = self.stage3(x)
        layer3_out = x
        x = self.stage4(x)
        layer4_out = x
        x = self.stage5(x)
        x = x.view(-1, self.num_classes)
        x = self.classifier(x)
        return x, layer1_out, layer2_out, layer3_out, layer4_out


model = Net()
if args.cuda:
    model.cuda()  # 将所有的模型参数移动到GPU上

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(
            target)  # Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        data = data.view(-1, 784)
        # print(target.size())
        optimizer.zero_grad()
        output, layer1_out, layer2_out, layer3_out, layer4_out = model(data)
        loss = F.nll_loss(output, target)  # 负log似然损失
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))



def test(epoch):
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 784)
        output, layer1_out, layer2_out, layer3_out, layer4_out = model(data)
        test_loss += F.nll_loss(output, target).data[0]  # Variable.data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_with_tsne(model_path):
    load_state(model_path, model)
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    data_all = Variable()
    first = True
    print('start test:')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if first :
            data_all, target_all = Variable(data, volatile=True), Variable(target)
            data_all = data_all.view(-1, 784)
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 784)
        data_all = torch.cat((data, data_all), 0)
        # print(data_all.size())

        output, layer1_out, layer2_out, layer3_out, layer4_out = model(data)
        test_loss += F.nll_loss(output, target).data[0]  # Variable.data
        if first:
            pred_all = output.data.max(1)[1]
            layer1_out_all = layer1_out
            layer2_out_all = layer2_out
            layer3_out_all = layer3_out
            layer4_out_all = layer4_out
            first = False

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pred_all = torch.cat((pred, pred_all), 0)
        layer1_out_all = torch.cat((layer1_out, layer1_out_all), 0)
        layer2_out_all = torch.cat((layer2_out, layer2_out_all), 0)
        layer3_out_all = torch.cat((layer3_out, layer3_out_all), 0)
        layer4_out_all = torch.cat((layer4_out, layer4_out_all), 0)
        # print(pred_all.size())
        correct += pred.eq(target.data).cpu().sum()
    # print(data_all.size())
    # print(pred_all.size())
    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    layer1_out_all = layer1_out_all.data.numpy()
    layer2_out_all = layer2_out_all.data.numpy()
    layer3_out_all = layer3_out_all.data.numpy()
    layer4_out_all = layer4_out_all.data.numpy()
    # data_all = pd.DataFrame(data_all, index=data_all[:, 0]),
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    layer1_out_all_tsne = np.array(tsne.fit_transform(layer1_out_all))[:, np.newaxis, :]
    layer2_out_all_tsne = np.array(tsne.fit_transform(layer2_out_all))[:, np.newaxis, :]
    layer3_out_all_tsne = np.array(tsne.fit_transform(layer3_out_all))[:, np.newaxis, :]
    layer4_out_all_tsne = np.array(tsne.fit_transform(layer4_out_all))[:, np.newaxis, :]

    layerout_tsne = layer1_out_all_tsne
    layerout_tsne = np.concatenate((layerout_tsne, layer2_out_all_tsne), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, layer3_out_all_tsne), axis=1)
    layerout_tsne = np.concatenate((layerout_tsne, layer4_out_all_tsne), axis=1)
    np.save('layerout_tsne.npy', layerout_tsne)
    # layerout_tsne = np.load('layerout_tsne.npy')
    # print(layerout_tsne.shape)
    # tsne = pd.DataFrame(tsne.embedding_, index=data_all.index)  # 转换数据格式

    colors = ['red', 'm', 'cyan', 'blue', 'lime', 'lawngreen', 'lightcoral', 'lightyellow', 'mediumorchid', 'mediumpurple']

    plt.figure(figsize=(10, 6))
    print('start plot:')
    for i in range(len(colors)):
        px = []
        py = []
        px2 = []
        py2 = []
        for j in range(1000):
            if pred_all[j] == i :
                plt.plot(layerout_tsne[j,:,0], layerout_tsne[j,:,1])
                # px.append(layerout_tsne[j, 0])
                # py.append(layerout_tsne[j, 1])

        # plt.scatter(px, py, s=20, c=colors[i], marker='o')
        # plt.scatter(px2, py2, s=20, c=colors[i], marker='v')

    # plt.legend(np.arange(0,5).astype(str))
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('C:/Users/Day/Desktop/PPT_report/Galaxy pic/Visualization/2/cnn1_train.png', dpi=300, bbox_inches='tight')
    plt.savefig('1.png', dpi=300,
                bbox_inches='tight')

    plt.show()


def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")

def load_state(load_path, model, optimizer=None):
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        # ckpt_keys = set(checkpoint['state_dict'].keys())
        # own_keys = set(model.state_dict().keys())
        # missing_keys = own_keys - ckpt_keys
        # for k in missing_keys:
        #     print('missing keys from checkpoint {}: {}'.format(load_path, k))
        pretrain(model, checkpoint['state_dict'])

        print("=> loaded model from checkpoint '{}'".format(load_path))
        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                  .format(load_path, start_epoch))
            return
    else:
        print("=> no checkpoint found at '{}'".format(load_path))

if __name__ == '__main__' :
    # for epoch in range(1, args.epochs + 1):
    #     train(epoch)
    #     test(epoch)
    # torch.save({
    #         'epoch': epoch ,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': 0,
    #         'optimizer': optimizer.state_dict(),
    #     }, '%s_%s.pth.tar' % (args.save_path, epoch))
    test_with_tsne(args.load_path)
