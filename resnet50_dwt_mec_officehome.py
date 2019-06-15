"""
File modified from:
	https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from __future__ import print_function

import sys
sys.path.append('utils')

import argparse
import os
import numpy as np
from PIL import Image
import scipy.io as sio
import cv2

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import batch_norm
import folder
import consensus_loss
import whitening

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class whitening_scale_shift(nn.Module):
	def __init__(self, planes, group_size, running_mean, running_variance, track_running_stats=True, affine=True):
		super(whitening_scale_shift, self).__init__()
		self.planes = planes
		self.group_size = group_size
		self.track_running_stats = track_running_stats
		self.affine = affine
		self.running_mean = running_mean
		self.running_variance = running_variance

		self.wh = whitening.WTransform2d(self.planes, 
										 self.group_size, 
										 running_m=self.running_mean, 
										 running_var=self.running_variance, 
										 track_running_stats=self.track_running_stats)
		if self.affine:
			self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
			self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))

	def forward(self, x):
		out = self.wh(x)
		if self.affine:
			out = out * self.gamma + self.beta
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, layer, sub_layer, bn_dict, group_size=4, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		self.conv1 = conv1x1(inplanes, planes)
		if layer == 1:
			self.bns1 = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_variance'],
											  affine=False)
			self.bnt1 = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_variance'],
											  affine=False)
			self.bnt1_aug = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.wh.running_variance'],
											  affine=False)
			self.gamma1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.gamma'])
			self.beta1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.beta'])
		else:
			self.bns1 = batch_norm.BatchNorm2d(num_features=planes,
											  running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
											  running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
											  affine=False)
			self.bnt1 = batch_norm.BatchNorm2d(num_features=planes,
											  running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
											  running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
											  affine=False)
			self.bnt1_aug = batch_norm.BatchNorm2d(num_features=planes,
											  running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_mean'],
											  running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.running_var'],
											  affine=False)
			self.gamma1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.weight'].view(-1, 1, 1))
			self.beta1 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.bias'].view(-1, 1, 1))

		self.conv2 = conv3x3(planes, planes, stride)
		if layer == 1:
			self.bns2 = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_variance'],
											  affine=False)
			self.bnt2 = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_variance'],
											  affine=False)
			self.bnt2_aug = whitening_scale_shift(planes=planes, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.wh.running_variance'],
											  affine=False)
			self.gamma2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.gamma'])
			self.beta2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.beta'])
		else:
			self.bns2 = batch_norm.BatchNorm2d(num_features=planes,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
											   affine=False)
			self.bnt2 = batch_norm.BatchNorm2d(num_features=planes,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
											   affine=False)
			self.bnt2_aug = batch_norm.BatchNorm2d(num_features=planes,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.running_var'],
											   affine=False)
			self.gamma2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.weight'].view(-1, 1, 1))
			self.beta2 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.bias'].view(-1, 1, 1))

		self.conv3 = conv1x1(planes, planes * self.expansion)
		if layer == 1:
			self.bns3 = whitening_scale_shift(planes=planes * self.expansion, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_variance'],
											  affine=False)
			self.bnt3 = whitening_scale_shift(planes=planes * self.expansion, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_variance'],
											  affine=False)
			self.bnt3_aug = whitening_scale_shift(planes=planes * self.expansion, 
											  group_size=group_size,
											  running_mean=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_mean'],
											  running_variance=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.wh.running_variance'],
											  affine=False)
			self.gamma3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.gamma'])
			self.beta3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.beta'])
		else:
			self.bns3 = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
											   affine=False)
			self.bnt3 = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
											   affine=False)
			self.bnt3_aug = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
											   running_m=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_mean'],
											   running_v=bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.running_var'],
											   affine=False)
			self.gamma3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.weight'].view(-1, 1, 1))
			self.beta3 = nn.Parameter(bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.bias'].view(-1, 1, 1))
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		if self.downsample is not None:
			if layer == 1:
				self.downsample_bns = whitening_scale_shift(planes=planes * self.expansion, 
															group_size=group_size,
															running_mean=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_mean'],
															running_variance=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_variance'],
															affine=False)
				self.downsample_bnt = whitening_scale_shift(planes=planes * self.expansion, 
															group_size=group_size,
															running_mean=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_mean'],
															running_variance=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_variance'],
															affine=False)
				self.downsample_bnt_aug = whitening_scale_shift(planes=planes * self.expansion, 
															group_size=group_size,
															running_mean=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_mean'],
															running_variance=bn_dict['layer' + str(layer) + '.0.downsample_bn.wh.running_variance'],
															affine=False)
				self.downsample_gamma = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample_bn.gamma'])
				self.downsample_beta = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample_bn.beta'])
			else:
				self.downsample_bns = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
															 running_m=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_mean'],
															 running_v=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_var'],
															 affine=False)
				self.downsample_bnt = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
															 running_m=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_mean'],
															 running_v=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_var'],
															 affine=False)
				self.downsample_bnt_aug = batch_norm.BatchNorm2d(num_features=planes * self.expansion,
															 running_m=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_mean'],
															 running_v=bn_dict['layer' + str(layer) + '.0.downsample_bn.running_var'],
															 affine=False)
				self.downsample_gamma = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample_bn.weight'].view(-1, 1, 1))
				self.downsample_beta = nn.Parameter(bn_dict['layer' + str(layer) + '.0.downsample_bn.bias'].view(-1, 1, 1))

	def forward(self, x):
		if self.training:
			# to do
			identity = x
			out = self.conv1(x)
			out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
			out = torch.cat((self.bns1(out_s), torch.cat((self.bnt1(out_t), self.bnt1_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma1 + self.beta1
			out = self.relu(out)

			out = self.conv2(out)
			out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
			out = torch.cat((self.bns2(out_s), torch.cat((self.bnt2(out_t), self.bnt2_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma2 + self.beta2
			out = self.relu(out)

			out = self.conv3(out)
			out_s, out_t, out_t_dup = torch.split(out, split_size_or_sections=out.shape[0] // 3, dim=0)
			out = torch.cat((self.bns3(out_s), torch.cat((self.bnt3(out_t), self.bnt3_aug(out_t_dup)), dim=0) ), dim=0) * self.gamma3 + self.beta3

			if self.downsample is not None:
				identity = self.downsample(x)
				identity_s, identity_t, identity_t_dup = torch.split(identity, split_size_or_sections=identity.shape[0] // 3, dim=0)
				identity = torch.cat((self.downsample_bns(identity_s), 
					torch.cat((self.downsample_bnt(identity_t), self.downsample_bnt_aug(identity_t_dup)), dim=0) ), dim=0) * self.downsample_gamma + self.downsample_beta

			out = out.clone() + identity
			out = self.relu(out)
		else:
			identity = x

			out = self.conv1(x)
			out = self.bnt1(out) * self.gamma1 + self.beta1 
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bnt2(out) * self.gamma2 + self.beta2
			out = self.relu(out)

			out = self.conv3(out)
			out = self.bnt3(out) * self.gamma3 + self.beta3

			if self.downsample is not None:
				identity = self.downsample(x)
				identity = self.downsample_bnt(identity) * self.downsample_gamma + self.downsample_beta

			out = out.clone() + identity
			out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, state_dict, num_classes=65, zero_init_residual=False, group_size=4):
		super(ResNet, self).__init__()
		self.inplanes = 64
		self.bn_dict = compute_bn_stats(state_dict)

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bns1 = whitening_scale_shift(planes=64, 
										  group_size=group_size,
										  running_mean=self.bn_dict['bn1.wh.running_mean'],
										  running_variance=self.bn_dict['bn1.wh.running_variance'],
										  affine=False)
		self.bnt1 = whitening_scale_shift(planes=64, 
										  group_size=group_size,
										  running_mean=self.bn_dict['bn1.wh.running_mean'],
										  running_variance=self.bn_dict['bn1.wh.running_variance'],
										  affine=False)
		self.bnt1_aug = whitening_scale_shift(planes=64, 
										  group_size=group_size,
										  running_mean=self.bn_dict['bn1.wh.running_mean'],
										  running_variance=self.bn_dict['bn1.wh.running_variance'],
										  affine=False)
		self.gamma1 = nn.Parameter(self.bn_dict['bn1.gamma'])
		self.beta1 = nn.Parameter(self.bn_dict['bn1.beta'])

		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], self.bn_dict, layer=1)
		self.layer2 = self._make_layer(block, 128, layers[1], self.bn_dict, stride=2, layer=2)
		self.layer3 = self._make_layer(block, 256, layers[2], self.bn_dict, stride=2, layer=3)
		self.layer4 = self._make_layer(block, 512, layers[3], self.bn_dict, stride=2, layer=4)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc_out = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, bn_dict, layer=1, group_size=4, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				#nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, layer, 0, bn_dict, group_size, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, layer, i, bn_dict, group_size))

		return nn.Sequential(*layers)

	def forward(self, x):
		if self.training:
			x = self.conv1(x)
			x_s, x_t, x_t_dup = torch.split(x, split_size_or_sections=x.shape[0] // 3, dim=0)
			x = torch.cat((self.bns1(x_s), torch.cat((self.bnt1(x_t), self.bnt1_aug(x_t_dup)), dim=0) ), dim=0) * self.gamma1 + self.beta1
			x = self.relu(x)
			x = self.maxpool(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)

			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc_out(x)
		else:
			x = self.conv1(x)
			x = self.bnt1(x) * self.gamma1 + self.beta1
			x = self.relu(x)
			x = self.maxpool(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)

			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc_out(x)

		return x

def resnet50(weights_path, device):
	
	state_dict_ = torch.load(weights_path, map_location=device)
	state_dict_model = state_dict_['state_dict']
	
	modified_state_dict = {}
	for key in state_dict_model.keys():
		mod_key = key[7:]
		modified_state_dict.update({mod_key: state_dict_model[key]})

	model = ResNet(Bottleneck, [3, 4, 6, 3], modified_state_dict)
	model.load_state_dict(modified_state_dict, strict=False)
	
	return model

def eval_pass_collect_stats(args, model, device, target_test_loader):
	# Run a bunch of forward passes to collect the target statistics before evaluating on the test set
	model.train(mode=True)
	with torch.no_grad():
		for i in range(10):
			print("Pass {} ...".format(i))
			for data, _ in target_test_loader:
				data = torch.cat((data, data, data), dim=0)			# dont care about source statistics after its trained.
				data = data.to(device)
				output = model(data)

def train_infinite_collect_stats(args, model, device, source_train_loader,
								 target_train_loader, optimizer, lambda_mec_loss,
								 target_test_loader):

	source_iter = iter(source_train_loader)
	target_iter = iter(target_train_loader)

	exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6000], gamma=0.1)

	for i in range(args.num_iters):
		model.train()

		exp_lr_scheduler.step()
		try:
			source_data, source_y = next(source_iter)
		except:
			source_iter = iter(source_train_loader)
			source_data, source_y = next(source_iter)

		try:
			target_data, target_data_dup, _ = next(target_iter)
		except:
			target_iter = iter(target_train_loader)
			target_data, target_data_dup, _ = next(target_iter)

		data = torch.cat((source_data, target_data, target_data_dup), dim=0)		# concat the source and target mini-batches
		data, source_y = data.to(device), source_y.to(device)

		optimizer.zero_grad()
		output = model(data)
		source_output, target_output, target_output_dup = torch.split(output, split_size_or_sections=output.shape[0] // 3, dim=0)
		
		mec_criterion = consensus_loss.MinEntropyConsensusLoss(num_classes=args.num_classes, device=device)

		cls_loss = F.nll_loss(F.log_softmax(source_output), source_y)
		mec_loss = lambda_mec_loss * mec_criterion(target_output, target_output_dup)

		loss = cls_loss + mec_loss
		loss.backward()

		optimizer.step()

		if i % args.log_interval == 0:
			print('Train Iter: [{}/{}]\tClassification Loss: {:.6f} \t MEC Loss: {:.6f}'.format(
				i, args.num_iters, cls_loss.item(), mec_loss.item()
			))

		if (i + 1) % args.check_acc_step == 0:
			test(args, model, device, target_test_loader)

	print("Training is complete...")
	print("Running a bunch of forward passes to estimate the population statistics of target...")
	eval_pass_collect_stats(args, model, device, target_test_loader)
	print("Finally computing the precision on the test set...")
	test(args, model, device, target_test_loader)

def test(args, model, device, target_test_loader):
	model.eval()
	test_loss = 0.
	correct = 0
	with torch.no_grad():
		for data, target in target_test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).item()
			pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(target_test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(target_test_loader.dataset),
		100. * correct / len(target_test_loader.dataset)))

	return 100. * correct / len(target_test_loader.dataset)

def compute_bn_stats(state_dict):
	#state_dict = state_dict = torch.load(path) #'/home/sroy/.torch/models/resnet50-19c8e357.pth'

	bn_key_names = []
	for name, param in state_dict.items():
		if name.find('bn') != -1:
			bn_key_names.append(name)
		elif name.find('downsample') != -1:
			bn_key_names.append(name)

	# keeping only the batch norm specific elements in the dictionary
	bn_dict = {k: v for k, v in state_dict.items() if k in bn_key_names}

	return bn_dict

def _random_affine_augmentation(x):
	M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0], 
				[np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
	rows, cols = x.shape[1:3]
	dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols,rows))
	dst = np.transpose(dst, [2, 0, 1])
	return torch.from_numpy(dst)

def _gaussian_blur(x, sigma=0.1):
	ksize = int(sigma + 0.5) * 8 + 1
	dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
	return torch.from_numpy(dst)


def main():

	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch DWT-MEC OfficeHome')
	parser.add_argument('--num_workers', default=2, type=int)
	parser.add_argument('--source_batch_size', type=int, default=18, help='input source batch size for training (default: 20)')
	parser.add_argument('--target_batch_size', type=int, default=18, help='input target batch size for training (default: 20)')
	parser.add_argument('--test_batch_size', type=int, default=10, help='input batch size for testing (default: 10)')
	parser.add_argument('--s_dset_path', type=str, default='../data/OfficeHomeDataset_10072016/Art', help="The source dataset path")
	parser.add_argument('--t_dset_path', type=str, default='../data/OfficeHomeDataset_10072016/Clipart', help="The target dataset path")
	parser.add_argument('--resnet_path', type=str, default='../data/models/model_best_gr_4.pth.tar', help="The pre-trained model path")
	parser.add_argument('--img_resize', type=int, default=256, help='size of the input image')
	parser.add_argument('--img_crop_size', type=int, default=224, help='size of the cropped image')
	parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations to train (default: 10000)')
	parser.add_argument('--check_acc_step', type=int, default=100, help='number of iterations steps to check validation accuracy (default: 10)')
	parser.add_argument('--lr_change_step', type=int, default=1000)
	parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 0.01)')
	parser.add_argument('--num_classes', type=int, default=65, help='number of classes in the dataset')
	parser.add_argument('--sgd_momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
	parser.add_argument('--running_momentum', type=float, default=0.1, help='Running momentum for domain statistics(default: 0.1)')
	parser.add_argument('--lambda_mec_loss', type=float, default=0.1, help='Value of lambda for the entropy loss (default: 0.1)')
	parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

	args = parser.parse_args()

	# set the seed
	torch.manual_seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# transformation on the source data during training and test data during test
	data_transform = transforms.Compose([
			transforms.Resize((args.img_resize, args.img_resize)), # spatial size of vgg-f input
			transforms.RandomCrop((args.img_crop_size, args.img_crop_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# transformation on the target data
	data_transform_dup = transforms.Compose([
			transforms.Resize((args.img_resize, args.img_resize)),
			transforms.RandomCrop((args.img_crop_size, args.img_crop_size)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Lambda(lambda x: _random_affine_augmentation(x)),
			transforms.Lambda(lambda x: _gaussian_blur(x)),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


	# train data sets
	source_dataset = folder.ImageFolder(root=args.s_dset_path,
										transform=data_transform)
	target_dataset = folder.ImageFolder(root=args.t_dset_path, 
										transform=data_transform,
										transform_aug=data_transform_dup)

	# test data sets
	target_dataset_test = folder.ImageFolder(root=args.t_dset_path, 
											 transform=data_transform)

	# '''''''''''' Train loaders ''''''''''''''' #
	source_trainloader = torch.utils.data.DataLoader(source_dataset,
													 batch_size=args.source_batch_size, 
													 shuffle=True, 
													 num_workers=args.num_workers,
													 drop_last=True)

	target_trainloader = torch.utils.data.DataLoader(target_dataset,
													 batch_size=args.source_batch_size, 
													 shuffle=True, 
													 num_workers=args.num_workers,
													 drop_last=True)

	# '''''''''''' Test loader ''''''''''''''' #
	target_testloader = torch.utils.data.DataLoader(target_dataset_test,
													batch_size=args.test_batch_size, 
													shuffle=True, 
													num_workers=args.num_workers)

	model = resnet50(args.resnet_path, device).to(device)
	
	final_layer_params = []
	rest_of_the_net_params = []

	for name, param in model.named_parameters():
		if name.startswith('fc_out'):
			final_layer_params.append(param)
		else:
			rest_of_the_net_params.append(param)

	optimizer = optim.SGD([
		{'params': rest_of_the_net_params},
		{'params': final_layer_params, 'lr': args.lr}
	], lr=args.lr * 0.1, momentum=0.9, weight_decay=5e-4)


	train_infinite_collect_stats(args=args,
				   				 model=model,
				   				 device=device,
				   				 source_train_loader=source_trainloader,
				   				 target_train_loader=target_trainloader,
				   				 optimizer=optimizer,
				   				 lambda_mec_loss=args.lambda_mec_loss,
				   				 target_test_loader=target_testloader)

if __name__ == '__main__':
	main()