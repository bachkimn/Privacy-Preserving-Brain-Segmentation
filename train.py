import os
import numpy as np
import torch
import torch.utils.data as data
import argparse

from segmentation_loss import Dice_Loss
from segmentation import Unet3D as segnet
from autoencoder import Unet3D_encoder as encoder
from discriminator import discriminator
from ppmi import ppmi_pairs, onehot_tensor_to_segmap_numpy

from time import time
from metric import count_predictions

def train_epoch(models, optimizers, criterions, LAMBDA, train_set, batch_size,device):

	loader = data.DataLoader(train_set, batch_size = batch_size, num_workers = 16, pin_memory=True, shuffle = True)

	Encrypter = models['enc'].train()
	Segmentator = models['seg'].train()
	Discriminator = models['dis'].train()
	
	optimizer_es = optimizers['es']
	optimizer_d = optimizers['dis']
	
	segmentation_criterion = criterions['seg']
	discrimination_criterion = criterions['dis']
	
	run_seg_loss = 0
	run_adv_loss = 0
	run_dis_loss = 0
	
	TP, FP, TN, FN = 0, 0, 0, 0
	real_TP, real_FP, real_TN, real_FN = 0, 0, 0, 0
	fake_TP, fake_FP, fake_TN, fake_FN = 0, 0, 0, 0
	
	start_time = time()
	for step, (x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref) in enumerate(loader):
		'''train loop here'''
		x, x_ref, y, y_ref,d, d_p = x.to(device), x_ref.to(device), y.to(device), y_ref.to(device), d.to(device), d_p.to(device)
		
		'''Update segmentator and encoder'''
		optimizer_es.zero_grad()
		
		z = Encrypter(x)
		z_ref = Encrypter(x_ref)

		y_hat = Segmentator(z)
		y_hat_ref = Segmentator(z_ref)
		seg_loss_1 = segmentation_criterion(y_hat, y)
		seg_loss_2 = segmentation_criterion(y_hat_ref, y_ref)
		seg_loss = seg_loss_1 + seg_loss_2
		run_seg_loss += seg_loss.item()
		
		d_x_z = Discriminator(x,z)
		d_x_zref = Discriminator(x, z_ref)
		d_xref_z = Discriminator(x_ref,z)
		d_xref_zref = Discriminator(x_ref, z_ref)
		d_z_zref = Discriminator(z, z_ref)
		
		adv_loss = -0.2 * (discrimination_criterion(d_x_z, d_p) +\
						   discrimination_criterion(d_x_zref, d) +\
						   discrimination_criterion(d_xref_z, d) +\
						   discrimination_criterion(d_xref_zref, d_p) +\
						   discrimination_criterion(d_z_zref, d))
		
		run_adv_loss += adv_loss.item()
		
		loss = seg_loss + LAMBDA * adv_loss
		loss.backward()
		
		optimizer_es.step()
		
		'''Update discriminator'''
		optimizer_d.zero_grad()
		
		z = z.detach()
		z_ref = z_ref.detach()
		
		d_x_xref = Discriminator(x, x_ref)	
		d_x_z = Discriminator(x,z)
		d_x_zref = Discriminator(x, z_ref)
		d_xref_z = Discriminator(x_ref,z)
		d_xref_zref = Discriminator(x_ref, z_ref)
		d_z_zref = Discriminator(z, z_ref)
		
		dis_loss = (discrimination_criterion(d_x_xref, d) +\
					discrimination_criterion(d_x_z, d_p) +\
					discrimination_criterion(d_x_zref, d) +\
					discrimination_criterion(d_xref_z, d) +\
					discrimination_criterion(d_xref_zref, d_p) +\
					discrimination_criterion(d_z_zref, d)) / 6

		run_dis_loss += dis_loss.item()
		dis_loss.backward()
		optimizer_d.step()
		
		#print(step, seg_loss.item(), -adv_loss.item(),dis_loss.item())
		'''Count prediction'''
		TP, FP, TN, FN = count_predictions(d_x_xref, d)
		real_TP += TP
		real_FP += FP
		real_TN += TN
		real_FN += FN
		
		TP, FP, TN, FN = count_predictions(d_x_z, d_p)
		fake_TP += TP
		fake_FP += FP
		fake_TN += TN
		fake_FN += FN

		TP, FP, TN, FN = count_predictions(d_x_zref, d)
		fake_TP += TP
		fake_FP += FP
		fake_TN += TN
		fake_FN += FN

		TP, FP, TN, FN = count_predictions(d_xref_z, d)
		fake_TP += TP
		fake_FP += FP
		fake_TN += TN
		fake_FN += FN

		TP, FP, TN, FN = count_predictions(d_xref_zref, d_p)
		fake_TP += TP
		fake_FP += FP
		fake_TN += TN
		fake_FN += FN
		
		TP, FP, TN, FN = count_predictions(d_z_zref, d)
		fake_TP += TP
		fake_FP += FP
		fake_TN += TN
		fake_FN += FN
		
	dur = (time() - start_time)	
	seg_loss = run_seg_loss / (step + 1)
	adv_loss = run_adv_loss / (step + 1)
	dis_loss = run_dis_loss / (step + 1)
	dis_real_acc = (real_TP + real_TN) / (real_TP + real_FP + real_TN + real_FN)
	dis_fake_acc = (fake_TP + fake_TN) / (fake_TP + fake_FP + fake_TN + fake_FN)
	models = {'enc': Encrypter, 'seg': Segmentator, 'dis': Discriminator}
	optimizers = {'es': optimizer_es, 'dis': optimizer_d}
	
	print('|Train: ----------------------------')
	print('Seg_loss:{:.4f} | Adv_loss:{:.4f} | Dis_loss:{:.4f}'.format(seg_loss, -adv_loss, dis_loss))
	print('dis_real_acc:{:.4f} | TP:{} FP:{} TN:{} FN:{} '.format(dis_real_acc, real_TP, real_FP, real_TN, real_FN))
	print('dis_fake_acc:{:.4f} | TP:{} FP:{} TN:{} FN:{} '.format(dis_fake_acc, fake_TP, fake_FP, fake_TN, fake_FN))
	print('duration:{:.0f}'.format(dur))
	 
	return seg_loss, adv_loss, dis_loss,\
		   dis_real_acc, dis_fake_acc,\
		   models, optimizers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device')
	parser.add_argument('')

if __name__ == '__main__':
	main()
