import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets

import numpy as np
from tqdm import tqdm

def get_drops(layers):
    for layer in layers:
        if hasattr(layer, "drops"):
            print("drops: ", layer.drops)


class CtrlbDropout2D(nn.Module):
    def __init__(self):
        super(CtrlbDropout2D, self).__init__()       
        self.drops = 0
        self._iter = 0

    def _tensor_to_output(self,tensor):

      return torch.bernoulli(1-tensor)
    
    
    def _assembleCtrlb(self,x):
      # x: [B, C, H, W]
      x_pooled = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1])).squeeze() # x_pooled: [B, C]
      g = x_pooled**2
      orderedS = torch.abs(g)**0.5

      prob = (orderedS/(torch.max(orderedS,dim=1)[0].unsqueeze(1)))
      top_half, top_idx = torch.topk(prob, math.floor(0.1*prob.shape[1]),dim = 1, largest=True, sorted=False)
      btm_half, btm_idx = torch.topk(prob, math.floor(0.1*prob.shape[1]),dim = 1, largest=False,sorted=False)
      scalling = prob.gather(1,top_idx) - (prob.gather(1,top_idx) - prob.gather(1,btm_idx))

      prob = prob.scatter(1, top_idx, scalling)

      prob = torch.clamp(prob,0,1)
      
      return self._tensor_to_output(prob)

    def forward(self,x):
        H, W = x.shape[-2:]
        if not self.training:
            return x
        
        with torch.no_grad():
            drop = self._assembleCtrlb(x)
            drop = drop.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            self.drops = (self.drops + (torch.mean((torch.sum(drop, dim=1)/drop.shape[1])))) / 2.0
            print(self.drops)
        return x*drop



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use_ctrl_dropout=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.use_ctrl_dropout = use_ctrl_dropout
        self.ctrl_dp = CtrlbDropout2D()
        

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if not self.use_ctrl_dropout:
            # Use vanilla dropout
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
        else:
            # Use control dropout
            out = self.ctrl_dp(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use_ctrl_dropout=True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, use_ctrl_dropout)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use_ctrl_dropout):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use_ctrl_dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, use_ctrl_dropout=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use_ctrl_dropout)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use_ctrl_dropout)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use_ctrl_dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def cifar100(root_dir="./", augment=True, batch_size=64):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    print(logging + ' CIFAR 100.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root_dir+'data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root_dir+'data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes



n_epochs = 200
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.001
momentum = 0.5
log_interval = 100

random_seed = 1
torch.manual_seed(random_seed)


train_loader, test_loader, num_classes = cifar100(batch_size=batch_size_train)

network = WideResNet(
    depth=28,
    num_classes=num_classes,
    widen_factor=10, 
    dropRate=0.2,
    use_ctrl_dropout=False
).cuda()

optimizer = torch.optim.SGD(network.parameters(), 0.1, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120],gamma=0.2)


train_losses = []
train_correct_counter = []
train_total_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

tic = time.time()

for e in range(n_epochs):
    # Training phase    
    network.train()
    for idx, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()

        output = network(x)

        loss = F.cross_entropy(output, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        with torch.no_grad():
            predictions = torch.argmax(torch.softmax(output, dim=-1), dim=-1)

            num_correct = (predictions == y).sum()

            train_correct_counter.append(num_correct.item())
            train_total_counter.append(predictions.shape[0])

        if (idx+1) % log_interval == 0:
            toc = time.time()
            print("Epoch {}/{}".format(e+1, n_epochs))
            print("100 iteration in {:.3f} seconds".format(toc-tic))
            print("Train loss: {:.3f}".format(np.mean(train_losses[-log_interval:])))
            print("Train acc: {}/{} = {:.3f}".format(np.sum(train_correct_counter[-log_interval:]), np.sum(train_total_counter[-log_interval:]), np.sum(train_correct_counter[-log_interval:])/np.sum(train_total_counter[-log_interval:])))
            tic = time.time()
            print()
    
    # Testing phase
    network.eval()
    with torch.no_grad():
        test_correct_counter = 0
        test_total_counter = 0
        
        pbar = tqdm(enumerate(test_loader))
        for idx, (x, y) in pbar:
            x = x.cuda()
            y = y.cuda()

            output = network(x)

            predictions = torch.argmax(torch.softmax(output, dim=-1), dim=-1)

            num_correct = (predictions == y).sum()

            test_correct_counter += num_correct.item()
            test_total_counter += predictions.shape[0]
        print("="*40)
        print("Test acc: {:.3f} on epoch {}".format(test_correct_counter/test_total_counter, e+1))
        print("="*40)
    
torch.save(network.state_dict(), "../models/cifar-wrn-ctrlbdp.pth")



