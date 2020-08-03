from __future__ import print_function

import sys
sys.path.append('utils')
import argparse
import gzip
import os
import pickle
import urllib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from whitening import WTransform2d

usps_dataset_multiplier = 6

class USPS(data.Dataset):
    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, transform_aug=None, download=False):
        """Init the USPS data set"""
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of train = 7438, Num of test = 1860
        self.transform = transform
        self.transform_aug = transform_aug
        self.dataset_size = None

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset no found." +
                               " You can use download=True to download it.")
        self.train_data, self.train_labels = self.load_samples()

        if self.train:
            self.train_data = np.repeat(self.train_data, usps_dataset_multiplier, axis=0)
            self.train_labels = np.repeat(self.train_labels, usps_dataset_multiplier, axis=0)

            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0: usps_dataset_multiplier * self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0: usps_dataset_multiplier * self.dataset_size]]

        # self.train_data *= 255.0
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # NCHW

    def __getitem__(self, index):
        """ Get images and target labels for data loader

        Args:
            index (int): Index
        Returns:
            tuple (image, target): where target is the index of the class
        """

        img, label = self.train_data[index], self.train_labels[index]

        if self.transform_aug is not None:
            img_aug = self.transform_aug(img)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.squeeze(torch.LongTensor([np.int64(label).item()]))

        if self.transform_aug is not None:
            return img, img_aug, label
        else:
            return img, label

    def __len__(self):
        """ Return the size of the dataset """
        if self.train:
            return usps_dataset_multiplier * self.dataset_size
        else:
            return self.dataset_size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("Done")
        return

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()

        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, transform_aug=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform_aug = transform_aug
        self.train = train  # training set or test set

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target_label, rot_label) where target_label is index of the target class
            and rot_label is the rotation index
        """
        img, target_label = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform_aug is not None:
            img_aug = self.transform_aug(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.transform_aug is not None:
            return img, img_aug, target_label
        else:
            return img, target_label

    def __len__(self):
        return len(self.data)

class EntropyLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        p = F.softmax(x, dim=1)
        q = F.log_softmax(x, dim=1)
        b = p * q
        b = -1.0 * b.sum(-1).mean()
        #b = -1.0 * b.sum()
        return b

class LeNet(nn.Module):
    def __init__(self, group_size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.ws1 = WTransform2d(num_features=32, group_size=group_size)
        self.wt1 = WTransform2d(num_features=32, group_size=group_size)
        self.gamma1 = nn.Parameter(torch.ones(32, 1, 1))
        self.beta1 = nn.Parameter(torch.zeros(32, 1, 1))
        #self.conv1_drop = nn.Dropout2d()

        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, padding=2)
        self.ws2 = WTransform2d(num_features=48, group_size=group_size)
        self.wt2 = WTransform2d(num_features=48, group_size=group_size)
        self.gamma2 = nn.Parameter(torch.ones(48, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(48, 1, 1))
        #self.conv2_drop = nn.Dropout2d()

        self.fc3 = nn.Linear(2352, 100)
        self.bns3 = nn.BatchNorm1d(100, affine=False)
        self.bnt3 = nn.BatchNorm1d(100, affine=False)
        self.gamma3 = nn.Parameter(torch.ones(1, 100))
        self.beta3 = nn.Parameter(torch.zeros(1, 100))

        self.fc4 = nn.Linear(100, 100)
        self.bns4 = nn.BatchNorm1d(100, affine=False)
        self.bnt4 = nn.BatchNorm1d(100, affine=False)
        self.gamma4 = nn.Parameter(torch.ones(1, 100))
        self.beta4 = nn.Parameter(torch.zeros(1, 100))

        self.fc5 = nn.Linear(100, 10)
        self.bns5 = nn.BatchNorm1d(10, affine=False)
        self.bnt5 = nn.BatchNorm1d(10, affine=False)
        self.gamma5 = nn.Parameter(torch.ones(1, 10))
        self.beta5 = nn.Parameter(torch.zeros(1, 10))

    def forward(self, x):

        if self.training:
            x = self.conv1(x)
            x_source, x_target = torch.split(x, split_size_or_sections=x.shape[0] // 2, dim=0)
            #x = self.conv1_drop(F.max_pool2d(F.relu(torch.cat((self.ws1(x_source), self.wt1(x_target)), dim=0)*self.gamma1 + self.beta1), kernel_size=2, stride=2))
            x = F.max_pool2d(F.relu(torch.cat((self.ws1(x_source), self.wt1(x_target)), dim=0)*self.gamma1 + self.beta1), kernel_size=2, stride=2)

            x = self.conv2(x)
            x_source, x_target = torch.split(x, split_size_or_sections=x.shape[0] // 2, dim=0)
            #x = self.conv2_drop(F.max_pool2d(F.relu(torch.cat((self.ws2(x_source), self.wt2(x_target)), dim=0)*self.gamma2 + self.beta2), kernel_size=2, stride=2))
            x = F.max_pool2d(F.relu(torch.cat((self.ws2(x_source), self.wt2(x_target)), dim=0)*self.gamma2 + self.beta2), kernel_size=2, stride=2)

            x = x.view(x.shape[0], -1)
            x = self.fc3(x)
            x_source, x_target = torch.split(x, split_size_or_sections=x.shape[0] // 2, dim=0)
            #x = F.dropout(F.relu(torch.cat((self.bns3(x_source), self.bnt3(x_target)), dim=0)*self.gamma3 + self.beta3), training=self.training)
            x = F.relu(torch.cat((self.bns3(x_source), self.bnt3(x_target)), dim=0)*self.gamma3 + self.beta3)

            x = self.fc4(x)
            x_source, x_target = torch.split(x, split_size_or_sections=x.shape[0] // 2, dim=0)
            #x = F.dropout(F.relu(torch.cat((self.bns4(x_source), self.bnt4(x_target)), dim=0)*self.gamma4 + self.beta4), training=self.training)
            x = F.relu(torch.cat((self.bns4(x_source), self.bnt4(x_target)), dim=0)*self.gamma4 + self.beta4)

            x = self.fc5(x)
            x_source, x_target = torch.split(x, split_size_or_sections=x.shape[0] // 2, dim=0)
            x = torch.cat((self.bns5(x_source), self.bnt5(x_target)), dim=0)*self.gamma5 + self.beta5
        else:
            x = self.conv1(x)
            #x = self.conv1_drop(F.max_pool2d(F.relu(self.wt1(x)*self.gamma1 + self.beta1), kernel_size=2, stride=2))
            x = F.max_pool2d(F.relu(self.wt1(x)*self.gamma1 + self.beta1), kernel_size=2, stride=2)

            x = self.conv2(x)
            #x = self.conv2_drop(F.max_pool2d(F.relu(self.wt2(x)*self.gamma2 + self.beta2), kernel_size=2, stride=2))
            x = F.max_pool2d(F.relu(self.wt2(x)*self.gamma2 + self.beta2), kernel_size=2, stride=2)

            x = x.view(x.shape[0], -1)
            x = self.fc3(x)
            #x = F.dropout(F.relu(self.bnt3(x)*self.gamma3 + self.beta3), training=self.training)
            x = F.relu(self.bnt3(x)*self.gamma3 + self.beta3)

            x = self.fc4(x)
            #x = F.dropout(F.relu(self.bnt4(x)*self.gamma4 + self.beta4), training=self.training)
            x = F.relu(self.bnt4(x)*self.gamma4 + self.beta4)

            x = self.fc5(x)
            x = self.bnt5(x)*self.gamma5 + self.beta5
        return x


def train(args, model, device, source_train_loader, target_train_loader, optimizer, epoch, lambda_entropy_loss):
    model.train()
    for batch_idx, (source, target) in enumerate(zip(source_train_loader, target_train_loader)):
        source_data = source[0]
        source_y = source[1]
        target_data = target[0]

        data = torch.cat((source_data, target_data), dim=0)  # concat the source and target mini-batches
        data, source_y = data.to(device), source_y.to(device)

        optimizer.zero_grad()
        output = model(data)

        source_output, target_output = torch.split(output, split_size_or_sections=output.shape[0] // 2, dim=0)

        entropy_criterion = EntropyLoss()

        cls_loss = F.nll_loss(F.log_softmax(source_output), source_y)
        entropy_l = lambda_entropy_loss * entropy_criterion(target_output)

        loss = cls_loss + entropy_l
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tClassification Loss: {:.6f} \tEntropy Loss: {:.6f}'.format(
                epoch, batch_idx * len(target_data), len(source_train_loader.dataset),
                       100. * batch_idx / len(source_train_loader), cls_loss.item(), entropy_l.item()))

def test(args, model, device, target_test_loader):
    model.eval()
    test_cls_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_cls_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).item()
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_cls_loss /= len(target_test_loader.dataset)
        print('\nTest set: Classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_cls_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))

    return 100. * correct / len(target_test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DIAL example')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--source_batch_size', type=int, default=32, help='input source batch size for training (default: 32)')
    parser.add_argument('--target_batch_size', type=int, default=32, help='input target batch size for training (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=100, help='input batch size for testing (default: 100)')
    parser.add_argument('--source', type=str, default='usps', help='source dataset name')
    parser.add_argument('--target', type=str, default='mnist', help='target dataset name')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--sgd_momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--running_momentum', type=float, default=0.1, help='Running momentum for statistics(default: 0.1)')
    parser.add_argument('--lambda_entropy_loss', type=float, default=0.1, help='Value of lambda for the entropy loss (default: 0.1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--from_script', action='store_true', help="use this flag for bulk running from script")
    parser.add_argument('--run', default=0, type=int, help="use this flag for bulk running from script")
    parser.add_argument('--method', default='bn', help="use this flag for bulk running from script")
    parser.add_argument('--group_size', type=int, default=32, help='group size for the whitening matrix (default: 32)')
    args = parser.parse_args()
    assert args.source != args.target, "source and target datasets can not be the same"
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """ MNIST train and test data loaders """
    train_loader_mnist = torch.utils.data.DataLoader(
        MNIST('../data/mnist', train=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.1307], std=[0.3081])
              ])),
        batch_size=args.source_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    test_loader_mnist = torch.utils.data.DataLoader(
        MNIST('../data/mnist', train=False,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.1307], std=[0.3081])
              ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)

    """ USPS train and test data loaders """
    train_loader_usps = torch.utils.data.DataLoader(
        USPS(root='../data/usps', train=True,
             transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5], std=[0.5])
             ])
             ,download=True), batch_size=args.target_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    test_loader_usps = torch.utils.data.DataLoader(
        USPS(root='../data/usps', train=False,
             transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5], std=[0.5])
             ])
             ,download=False), batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers)

    model = LeNet(group_size=args.group_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    if args.source == 'mnist' and args.target == 'usps':
        source_train_loader = train_loader_mnist
        target_train_loader = train_loader_usps
        test_loader = test_loader_usps
    elif args.source == 'usps' and args.target == 'mnist':
        source_train_loader = train_loader_usps
        target_train_loader = train_loader_mnist
        test_loader = test_loader_mnist

    for epoch in range(args.epochs):
        exp_lr_scheduler.step()
        train(args, model, device, source_train_loader, target_train_loader, optimizer, epoch, args.lambda_entropy_loss)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()