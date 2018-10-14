from __future__ import print_function
import argparse
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

def batch_product(iput, mat2):
		result = None
		for i in range(iput.size()[0]):
			op = torch.mm(iput[i], mat2)
			op = op.unsqueeze(0)
			if(result is None):
				result = op
			else:
				result = torch.cat((result,op),0)
		return result.squeeze(2)


class rec_attention(nn.Module):
	# attention with bin context vector per HM and HM context vector
	def __init__(self,hm,args):
		super(rec_attention,self).__init__()
		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rep_size=args.bin_rnn_size*self.num_directions
		else:
			self.bin_rep_size=args.bin_rnn_size
	
		self.bin_context_vector=nn.Parameter(torch.Tensor(self.bin_rep_size,1),requires_grad=True)
	

		self.softmax=nn.Softmax()

		self.bin_context_vector.data.uniform_(-0.1, 0.1)

	def forward(self,iput):
		alpha=self.softmax(batch_product(iput,self.bin_context_vector))
		[batch_size,source_length,bin_rep_size2]=iput.size()
		repres=torch.bmm(alpha.unsqueeze(2).view(batch_size,-1,source_length),iput)
		return repres,alpha



class recurrent_encoder(nn.Module):
	# modular LSTM encoder
	def __init__(self,n_bins,ip_bin_size,hm,args):
		super(recurrent_encoder,self).__init__()
		self.bin_rnn_size=args.bin_rnn_size
		self.ipsize=ip_bin_size
		self.seq_length=n_bins

		self.num_directions=2 if args.bidirectional else 1
		if (hm==False):
			self.bin_rnn_size=args.bin_rnn_size
		else:
			self.bin_rnn_size=args.bin_rnn_size // 2
		self.bin_rep_size=self.bin_rnn_size*self.num_directions


		self.rnn=nn.LSTM(self.ipsize,self.bin_rnn_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional)

		self.bin_attention=rec_attention(hm,args)
	def outputlength(self):
		return self.bin_rep_size
	def forward(self,single_hm,hidden=None):

		bin_output, hidden = self.rnn(single_hm,hidden)
		bin_output = bin_output.permute(1,0,2)
		hm_rep,bin_alpha = self.bin_attention(bin_output)
		return hm_rep,bin_alpha

class raw_d(nn.Module):
	def __init__(self,args):
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		super(raw_d,self).__init__()
		self.rnn_hms=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.diffopsize=2*(self.opsize2)
		self.fdiff1_1=nn.Linear(self.opsize2,1)

	def forward(self,iput1,iput2):

		iput=iput1-iput2
		bin_a=None
		level1_rep=None
		[batch_size,_,_]=iput.size()

		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)

			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)
		level1_rep=level1_rep.permute(1,0,2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep)
		final_rep_1=final_rep_1.squeeze(1)
		prediction_m=((self.fdiff1_1(final_rep_1)))
		return prediction_m,hm_level_attention_1, bin_a


class raw_c(nn.Module):
	def __init__(self,args):
		super(raw_c,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.joint=False
		self.rnn_hms=nn.ModuleList()
		for i in range(2*self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(2*self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.diffopsize=2*(self.opsize2)
		self.fdiff1_1=nn.Linear(self.opsize2,1)

	def forward(self,iput1,iput2):
		iput=torch.cat((iput1,iput2),2)
		bin_a=None
		level1_rep=None
		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()

			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)

			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)

		level1_rep=level1_rep.permute(1,0,2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep)
		final_rep_1=final_rep_1.squeeze(1)
		prediction_m=((self.fdiff1_1(final_rep_1)))

		return prediction_m,hm_level_attention_1, bin_a


class raw(nn.Module):
	# Model with all raw features: difference and absolute features
	def __init__(self,args):
		super(raw,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.rnn_hms=nn.ModuleList()
		for i in range(3*self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(3*self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.diffopsize=2*(self.opsize2)
		self.fdiff1_1=nn.Linear(self.opsize2,1)

	def forward(self,iput1,iput2):
		iput3=iput1-iput2
		iput4=torch.cat((iput1,iput2),2)
		iput=(torch.cat((iput4,iput3),2))

		bin_a=None
		level1_rep=None
		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)

		level1_rep=level1_rep.permute(1,0,2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep)
		final_rep_1=final_rep_1.squeeze(1)

		prediction_m=((self.fdiff1_1(final_rep_1)))
		return prediction_m,hm_level_attention_1, bin_a


class aux(nn.Module):
	def __init__(self,args):
		super(aux,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.joint=False
		self.shared=False
		self.rnn_hms=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.rnn_hms2=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms2.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_2=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.f1_1=nn.Linear(self.opsize2,1)
		self.f2_1=nn.Linear(self.opsize2,1)
		self.diffopsize=2*(self.opsize2)
		self.predictor_1=nn.Linear(self.diffopsize,self.diffopsize//2)
		self.predictor_=nn.Linear(self.diffopsize//2,1)
		self.relu=nn.ReLU()

	def forward_once(self,iput,shared,cellid):
		bin_a=None
		level1_rep=None
		if(shared or cellid==1):
			for hm,hm_encdr in enumerate(self.rnn_hms):
				hmod=iput[:,:,hm].contiguous()
				hmod=torch.t(hmod).unsqueeze(2)
				op,a= hm_encdr(hmod)
				if level1_rep is None:
					level1_rep=op
					bin_a=a
				else:
					level1_rep=torch.cat((level1_rep,op),1)
					bin_a=torch.cat((bin_a,a),1)
			level1_rep=level1_rep.permute(1,0,2)
		else:
			for hm,hm_encdr in enumerate(self.rnn_hms2):

				hmod=iput[:,:,hm].contiguous()
				hmod=torch.t(hmod).unsqueeze(2)
				op,a= hm_encdr(hmod)
				if level1_rep is None:
					level1_rep=op
					bin_a=a
				else:
					level1_rep=torch.cat((level1_rep,op),1)
					bin_a=torch.cat((bin_a,a),1)
			level1_rep=level1_rep.permute(1,0,2)
		return level1_rep,bin_a

	def forward(self,iput1,iput2):
 
		level1_rep1,bin_a1=self.forward_once(iput1,self.shared,1)
		level1_rep2,bin_a2=self.forward_once(iput2,self.shared,2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep1)
		if(self.joint):
			final_rep_2,hm_level_attention_2=self.hm_level_rnn_1(level1_rep2)
		else:
			final_rep_2,hm_level_attention_2=self.hm_level_rnn_2(level1_rep2)
		final_rep_1=final_rep_1.squeeze(1)
		final_rep_2=final_rep_2.squeeze(1)
		prediction1=((self.f1_1(final_rep_1)))
		prediction2=((self.f2_1(final_rep_2)))

		mlp_input=torch.cat((final_rep_1,final_rep_2),1)
		mlp_input=self.relu(self.predictor_1(mlp_input))
		prediction=(self.predictor_(mlp_input))
		hm_level_attention=torch.cat((hm_level_attention_1,hm_level_attention_2),1)
		bin_a=torch.cat((bin_a1,bin_a2),1)

		return prediction,hm_level_attention,bin_a,prediction1,prediction2


class raw_aux(nn.Module):
	# level 1 takes raw feaures, level 2 takes aux features as well as raw feature embeddings from level 1
	# returns attention scores from raw part only
	def __init__(self,args):
		super(raw_aux,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.joint=False
		self.shared=False
		self.rnn_hms=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.rnn_hms3=nn.ModuleList()
		for i in range(3*self.n_hms):
			self.rnn_hms3.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.rnn_hms2=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms2.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_3=recurrent_encoder(5*self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_2=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.f1_1=nn.Linear(self.opsize2,1)
		self.f2_1=nn.Linear(self.opsize2,1)
		self.diffopsize=3*(self.opsize2)
		self.predictor_=nn.Linear(self.opsize2,1)
		self.relu=nn.ReLU()


	def forward(self,iput1,iput2):
		iput3=iput1-iput2
		iput4=torch.cat((iput1,iput2),2)
		iput=(torch.cat((iput4,iput3),2))
		bin_a=None
		level1_rep=None
		[batch_size,_,_]=iput.size()
		for hm,hm_encdr in enumerate(self.rnn_hms3):

			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)
		level1_rep=level1_rep.permute(1,0,2)

		bin_a1=None
		level1_rep1=None
		[batch_size,_,_]=iput1.size()
		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput1[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep1 is None:
				level1_rep1=op
				bin_a1=a
			else:
				level1_rep1=torch.cat((level1_rep1,op),1)
				bin_a1=torch.cat((bin_a1,a),1)

		level1_rep1=level1_rep1.permute(1,0,2)
		bin_a2=None
		level1_rep2=None

		for hm,hm_encdr in enumerate(self.rnn_hms2):
			hmod=iput2[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep2 is None:
				level1_rep2=op
				bin_a2=a
			else:
				level1_rep2=torch.cat((level1_rep2,op),1)
				bin_a2=torch.cat((bin_a2,a),1)
		level1_rep2=level1_rep2.permute(1,0,2)
		level1_rep3=torch.cat((level1_rep,level1_rep1,level1_rep2),0)
		final_rep_3,hm_level_attention_3=self.hm_level_rnn_3(level1_rep3)
		final_rep_3=final_rep_3.squeeze(1)

		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep1)
		final_rep_2,hm_level_attention_2=self.hm_level_rnn_2(level1_rep2)
		final_rep_1=final_rep_1.squeeze(1)
		final_rep_2=final_rep_2.squeeze(1)
		prediction1=((self.f1_1(final_rep_1)))
		prediction2=((self.f2_1(final_rep_2)))
		prediction=(self.predictor_(final_rep_3))
		return prediction,hm_level_attention_3,bin_a,prediction1,prediction2


class aux_siamese(nn.Module):
	# aux with siamese  same as aux model mostly, but with shared level 1 embedding
	def __init__(self,args):
		super(aux_siamese,self).__init__()
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.joint=False
		self.shared=True
		self.rnn_hms=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_2=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.f1_1=nn.Linear(self.opsize2,1)
		self.f2_1=nn.Linear(self.opsize2,1)
		self.diffopsize=2*(self.opsize2)
		self.predictor_1=nn.Linear(self.diffopsize,self.diffopsize//2)
		self.predictor_=nn.Linear(self.diffopsize//2,1)
		self.relu=nn.ReLU()

	def forward_once(self,iput):
		bin_a=None
		level1_rep=None
		[batch_size,_,_]=iput.size()
		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)
		level1_rep=level1_rep.permute(1,0,2)

		return level1_rep,bin_a


	def forward(self,iput1,iput2):
		level1_rep1,bin_a1=self.forward_once(iput1)

		level1_rep2,bin_a2=self.forward_once(iput2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep1)
		final_rep_2,hm_level_attention_2=self.hm_level_rnn_2(level1_rep2)
		final_rep_1=final_rep_1.squeeze(1)
		final_rep_2=final_rep_2.squeeze(1)
		[a,b,c]=(level1_rep1.size())
		alpha_rep_1=level1_rep1.permute(1,0,2).view(b,a*c)
		alpha_rep_2=level1_rep2.permute(1,0,2).view(b,a*c)
		prediction1=((self.f1_1(final_rep_1)))
		prediction2=((self.f2_1(final_rep_2)))
		mlp_input=torch.cat((final_rep_1,final_rep_2),1)
		mlp_input=self.relu(self.predictor_1(mlp_input))
		prediction=(self.predictor_(mlp_input))
		hm_level_attention=torch.cat((hm_level_attention_1,hm_level_attention_2),1)
		bin_a=torch.cat((bin_a1,bin_a2),1)

		return prediction,hm_level_attention,bin_a,alpha_rep_1,alpha_rep_2,prediction1,prediction2






class raw_aux_siamese(nn.Module):
	# similar to raw_aux model with shared level 1 embedding
	# returns only raw level attentions
	# returns embeddings from shared level 1 for contrastive loss
	def __init__(self,args):
		self.n_hms=args.n_hms
		self.n_bins=args.n_bins
		self.ip_bin_size=1
		self.joint=False
		self.shared=True
		super(raw_aux_siamese,self).__init__()
		self.rnn_hms=nn.ModuleList()
		self.rnn_hmsx=nn.ModuleList()
		for i in range(self.n_hms):
			self.rnn_hms.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		for i in range(3*self.n_hms):
			self.rnn_hmsx.append(recurrent_encoder(self.n_bins,self.ip_bin_size,False,args))
		self.opsize = self.rnn_hms[0].outputlength()
		self.hm_level_rnn_1=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_1x=recurrent_encoder(5*self.n_hms,self.opsize,True,args)
		self.hm_level_rnn_2=recurrent_encoder(self.n_hms,self.opsize,True,args)
		self.opsize2=self.hm_level_rnn_1.outputlength()
		self.f1_1=nn.Linear(self.opsize2,1)
		self.f2_1=nn.Linear(self.opsize2,1)
		self.diffopsize=3*(self.opsize2)
		#self.predictor_1=nn.Linear(self.diffopsize,self.diffopsize//2)
		self.predictor_=nn.Linear(self.opsize2,1)
		self.relu=nn.ReLU()
		self.finalsoftmax=nn.LogSoftmax()
	def forward_once(self,iput):
		bin_a=None
		level1_rep=None
		[batch_size,_,_]=iput.size()
		for hm,hm_encdr in enumerate(self.rnn_hms):
			hmod=iput[:,:,hm].contiguous()
			hmod=torch.t(hmod).unsqueeze(2)
			op,a= hm_encdr(hmod)
			if level1_rep is None:
				level1_rep=op
				bin_a=a
			else:
				level1_rep=torch.cat((level1_rep,op),1)
				bin_a=torch.cat((bin_a,a),1)
		level1_rep=level1_rep.permute(1,0,2)
		return level1_rep,bin_a


	def forward(self,iput1,iput2):
		iput3=iput1-iput2
		iput4=torch.cat((iput1,iput2),2)
		iput=(torch.cat((iput4,iput3),2))
		bin_ax=None
		level1_repx=None
		[batch_size,_,_]=iput.size()
		for hm,hm_encdr in enumerate(self.rnn_hmsx):
			hmodx=iput[:,:,hm].contiguous()
			hmodx=torch.t(hmodx).unsqueeze(2)
			opx,ax= hm_encdr(hmodx)
			if level1_repx is None:
				level1_repx=opx
				bin_ax=ax
			else:
				level1_repx=torch.cat((level1_repx,opx),1)
				bin_ax=torch.cat((bin_ax,ax),1)
		level1_repx=level1_repx.permute(1,0,2)

		level1_rep1,bin_a1=self.forward_once(iput1)

		level1_rep2,bin_a2=self.forward_once(iput2)
		final_rep_1,hm_level_attention_1=self.hm_level_rnn_1(level1_rep1)
		final_rep_2,hm_level_attention_2=self.hm_level_rnn_2(level1_rep2)
		final_rep_1=final_rep_1.squeeze(1)
		final_rep_2=final_rep_2.squeeze(1)
		[a,b,c]=(level1_rep1.size())
		alpha_rep_1=level1_rep1.permute(1,0,2).view(b,a*c)
		alpha_rep_2=level1_rep2.permute(1,0,2).view(b,a*c)
		level1_rep3=torch.cat((level1_repx,level1_rep1,level1_rep2),0)
		final_rep_1x,hm_level_attention_1x=self.hm_level_rnn_1x(level1_rep3)
		final_rep_1x=final_rep_1x.squeeze(1)
		prediction1=((self.f1_1(final_rep_1)))
		prediction2=((self.f2_1(final_rep_2)))
		prediction=(self.predictor_(final_rep_1x))

		return prediction,hm_level_attention_1x,bin_ax,alpha_rep_1,alpha_rep_2,prediction1,prediction2
