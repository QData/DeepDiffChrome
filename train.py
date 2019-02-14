import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import matplotlib
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
import models as Model
from SiameseLoss import ContrastiveLoss
import evaluate
import data
import gc
import csv

parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--model_name', type=str, default='raw_d', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=90, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--cell_1', type=str, default='Cell1', help='cell type 1')
parser.add_argument('--cell_2', type=str, default='Cell2', help='cell type 2')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--data_root', type=str, default='./data/', help='data location')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=200, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_on_saved_model',action='store_true', help='only test on saved model')
args = parser.parse_args()

torch.manual_seed(1)



model_name = ''
model_name += (args.cell_1)+('_')+(args.cell_2)+('_')

model_name+=args.model_name




args.bidirectional=not args.unidirectional

print('the model name: ',model_name)
args.data_root+=''
args.save_root+=''
args.dataset=args.cell_1+('_')+args.cell_2
args.data_root = os.path.join(args.data_root)
print('loading data from:  ',args.data_root)
args.save_root = os.path.join(args.save_root,args.dataset)
print('saving results in  from: ',args.save_root)
model_dir = os.path.join(args.save_root,model_name)
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
attentionmapfile=model_dir+'/'+args.attentionfilename
print('==>processing data')
Train,Valid,Test = data.load_data(args)







CON=False
AUX=False
print('==>building model')
if(args.model_name=='raw_d'):
	model = Model.raw_d(args)
elif(args.model_name=='raw_c'):
	model = Model.raw_c(args)
elif(args.model_name=='raw'):
	model = Model.raw(args)
elif(args.model_name=='aux'):
	args.shared=False
	model = Model.aux(args)
	AUX=True
	args.gamma=0.0
elif(args.model_name=='raw_aux'):
	args.shared=False
	model = Model.raw_aux(args)
	AUX=True
	args.gamma=0.0
elif(args.model_name=='aux_siamese'):
	CON=True
	args.shared=True
	model = Model.aux_siamese(args)
	AUX=True
	args.gamma=4.0
elif(args.model_name=='raw_aux_siamese'):
	CON=True
	args.shared=True
	model = Model.raw_aux_siamese(args)
	AUX=True
	args.gamma=4.0
else:
	sys.exit("invalid model name")


if torch.cuda.device_count() > 1:
	torch.cuda.manual_seed_all(1)
	dtype = torch.cuda.FloatTensor
	cuda.set_device(args.gpuid)
	model.type(dtype)
	print('Using GPU '+str(args.gpuid))
else:
	dtype = torch.FloatTensor

print(model)
if(args.test_on_saved_model==False):
	print("==>initializing a new model")
	for p in model.parameters():
		p.data.uniform_(-0.1,0.1)

DiffLoss = nn.MSELoss(size_average=True).type(dtype)
AuxLoss = nn.MSELoss(size_average=True).type(dtype)
ConLoss = ContrastiveLoss().type(dtype)

optimizer = optim.Adam(model.parameters(), lr = args.lr)
#optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
def train(TrainData):
	model.train()
	# initialize attention
	diff_targets = torch.zeros(TrainData.dataset.__len__(),1)
	diff_predictions = torch.zeros(diff_targets.size(0),1)
	if(args.model_name=='raw_d'):
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)
	elif(args.model_name=='raw_c'):
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(2*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),2*args.n_hms)
	elif(args.model_name=='raw'):
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(3*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),3*args.n_hms)

	elif(args.model_name=='aux' or args.model_name=='aux_siamese'):
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(2*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),2*args.n_hms)

	elif(args.model_name=='raw_aux' or args.model_name=='raw_aux_siamese'):
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(3*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),5*args.n_hms)

	else:
		all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)

	num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
	all_gene_ids=[None]*TrainData.dataset.__len__()
	per_epoch_loss = 0
	for idx, Sample in enumerate(TrainData):
		if(idx%100==0):
			print('TRAINING ON BATCH:',idx)
		start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())
		optimizer.zero_grad()
		# get HM profiles
		inputs_1 = Sample['X_A']
		inputs_2 = Sample['X_B']
		# get targets: both differential and cell specific expression
		batch_diff_targets=(Sample['diff']).float().unsqueeze(1)
		batch_diff_targets_c1=(Sample['abs_A']).float().unsqueeze(1)
		batch_diff_targets_c2=(Sample['abs_B']).float().unsqueeze(1)
		diff_targets[start:end,0] = batch_diff_targets[:,0]

		if(CON==True):
			# get labels for contrastive loss
			batch_contrastive_targets =[]
			for label in batch_diff_targets:
				if(label<=-2.0):
					batch_contrastive_targets.append(1)
				elif(label>=2.0):
					batch_contrastive_targets.append(1)
				else:
					batch_contrastive_targets.append(0)
			batch_contrastive_targets=torch.Tensor(batch_contrastive_targets)


		all_gene_ids[start:end]=Sample['geneID']
		batch_size = inputs_1.size(0)

		if(AUX==False):
			# for raw models: raw_d, raw_c, raw
			batch_diff_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype),inputs_2.type(dtype))
			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
		elif(CON==False):
			# for aux models
			batch_diff_predictions,batch_beta,batch_alpha,batch_diff_predictions_c1,batch_diff_predictions_c2 = model(inputs_1.type(dtype),inputs_2.type(dtype))
			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c1,batch_diff_targets_c1.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c2,batch_diff_targets_c2.type(dtype))
		else:
			# for aux and siamese models
			batch_diff_predictions,batch_beta,batch_alpha,embedding_1,embedding_2,batch_diff_predictions_c1,batch_diff_predictions_c2 = model(inputs_1.type(dtype),inputs_2.type(dtype))

			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c1,batch_diff_targets_c1.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c2,batch_diff_targets_c2.type(dtype))
			loss+=args.gamma*ConLoss(embedding_1,embedding_2,batch_contrastive_targets.type(dtype))

		diff_predictions[start:end] = batch_diff_predictions.data.cpu()
		per_epoch_loss += loss.item()
		loss.backward()
		torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
		optimizer.step()
	per_epoch_loss=per_epoch_loss/num_batches
	return diff_predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids



def test(ValidData):
	model.eval()

	diff_targets = torch.zeros(ValidData.dataset.__len__(),1)
	diff_predictions = torch.zeros(diff_targets.size(0),1)
	if(args.model_name=='raw_d'):
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)
	elif(args.model_name=='raw_c'):
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(2*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),2*args.n_hms)
	elif(args.model_name=='raw'):
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(3*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),3*args.n_hms)
	elif(args.model_name=='aux' or args.model_name=='aux_siamese'):
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(2*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),2*args.n_hms)
	elif(args.model_name=='raw_aux' or args.model_name=='raw_aux_siamese'):
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(3*args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),5*args.n_hms)
	else:
		all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
		all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

	num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
	all_gene_ids=[None]*ValidData.dataset.__len__()
	per_epoch_loss = 0
	for idx, Sample in enumerate(ValidData):
		if(idx%100==0):
			print('TESTING ON BATCH:',idx)
		start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
		optimizer.zero_grad()
		# get HM profiles
		inputs_1 = Sample['X_A']
		inputs_2 = Sample['X_B']
		# get targets: both differential and cell specific expression
		batch_diff_targets=(Sample['diff']).float().unsqueeze(1)
		batch_diff_targets_c1=(Sample['abs_A']).float().unsqueeze(1)
		batch_diff_targets_c2=(Sample['abs_B']).float().unsqueeze(1)
		diff_targets[start:end,0] = batch_diff_targets[:,0]

		if(CON==True):
			# get labels for contrastive loss
			batch_contrastive_targets =[]
			for label in batch_diff_targets:
				if(label<=-2.0):
					batch_contrastive_targets.append(1)
				elif(label>=2.0):
					batch_contrastive_targets.append(1)
				else:
					batch_contrastive_targets.append(0)
			batch_contrastive_targets=torch.Tensor(batch_contrastive_targets)


		all_gene_ids[start:end]=Sample['geneID']
		batch_size = inputs_1.size(0)

		if(AUX==False):
			# for raw models: raw_d, raw_c, raw
			batch_diff_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype),inputs_2.type(dtype))
			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
		elif(CON==False):
			# for aux models
			batch_diff_predictions,batch_beta,batch_alpha,batch_diff_predictions_c1,batch_diff_predictions_c2 = model(inputs_1.type(dtype),inputs_2.type(dtype))
			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c1,batch_diff_targets_c1.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c2,batch_diff_targets_c2.type(dtype))
		else:
			# for aux and siamese models
			batch_diff_predictions,batch_beta,batch_alpha,embedding_1,embedding_2,batch_diff_predictions_c1,batch_diff_predictions_c2 = model(inputs_1.type(dtype),inputs_2.type(dtype))

			all_attention_bin[start:end]=batch_alpha.data
			all_attention_hm[start:end]=batch_beta.data
			loss = DiffLoss(batch_diff_predictions,batch_diff_targets.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c1,batch_diff_targets_c1.type(dtype))
			loss+=AuxLoss(batch_diff_predictions_c2,batch_diff_targets_c2.type(dtype))
			loss+=args.gamma*ConLoss(embedding_1,embedding_2,batch_contrastive_targets.type(dtype))

		diff_predictions[start:end] = batch_diff_predictions.data.cpu()
		per_epoch_loss += loss.item()
	per_epoch_loss=per_epoch_loss/num_batches
	return diff_predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids







best_valid_loss = 10000000000
best_valid_MSE=100000
best_valid_R2=-1
if(args.test_on_saved_model==False):
	for epoch in range(0, args.epochs):
		print('=---------------------------------------- Training '+str(epoch+1)+' -----------------------------------=')
		diff_predictions,diff_targets,alpha_train,beta_train,train_loss,_ = train(Train)
		train_MSE, train_R2 = evaluate.compute_metrics(diff_predictions,diff_targets)
		diff_predictions,diff_targets,alpha_valid,beta_valid,valid_loss,gene_ids_valid = test(Valid)
		valid_MSE, valid_R2 = evaluate.compute_metrics(diff_predictions,diff_targets)

		if(valid_R2 >= best_valid_R2):
				# save best epoch -- models converge early
			best_valid_R2=valid_R2
			torch.save(model,model_dir+"/"+model_name+'_R2_model.pt')

		print("Epoch:",epoch)
		print("train R2:",train_R2)
		print("valid R2:",valid_R2)
		print("best valid R2:", best_valid_R2)

 
	print("finished training!!")
	print("best validation R2:",best_valid_R2)
	print("testing")
	model=torch.load(model_dir+"/"+model_name+'_R2_model.pt')

	diff_predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test)
	test_MSE, test_R2 = evaluate.compute_metrics(diff_predictions,diff_targets)
	print("test R2:",test_R2)

	if(args.save_attention_maps):
		attentionfile=open(attentionmapfile,'w')
		attentionfilewriter=csv.writer(attentionfile)
		beta_test=beta_test.numpy()
		for i in range(len(gene_ids_test)):
			gene_attention=[]
			gene_attention.append(gene_ids_test[i])
			for e in beta_test[i,:]:
				gene_attention.append(str(e))
			attentionfilewriter.writerow(gene_attention)
		attentionfile.close()


else:
	model=torch.load(model_dir+"/"+model_name+'_R2_model.pt')
	diff_predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test = test(Test)
	test_MSE, test_R2 = evaluate.compute_metrics(diff_predictions,diff_targets)
	print("test R2:",test_R2)

	if(args.save_attention_maps):
		attentionfile=open(attentionmapfile,'w')
		attentionfilewriter=csv.writer(attentionfile)
		beta_test=beta_test.numpy()
		for i in range(len(gene_ids_test)):
			gene_attention=[]
			gene_attention.append(gene_ids_test[i])
			for e in beta_test[i,:]:
				gene_attention.append(str(e))
			attentionfilewriter.writerow(gene_attention)
		attentionfile.close()