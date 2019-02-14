import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import evaluate
import data
import gc

class ContrastiveLoss(torch.nn.Module):
	# Contrastive Loss formulation.
	#Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2)
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive
