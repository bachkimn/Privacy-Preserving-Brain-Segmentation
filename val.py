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
from metric import count_predictions, compute_metric, compute_dice_score

def val_epoch(models, criterions, val_set, batch_size, device):
	
	loader = data.DataLoader(val_set, batch_size = batch_size, num_workers = 16, pin_memory=True, shuffle = True)
	
	Encrypter = models['enc'].eval()
	Segmentator = models['seg'].eval()
	Discriminator = models['dis'].eval()
	
	segmentation_criterion = criterions['seg']
	discrimination_criterion = criterions['dis']
	
	run_seg_loss = 0
	run_adv_loss = 0
	run_dis_loss = 0
	dice_score = np.zeros(6)

	
	TP, FP, TN, FN = 0, 0, 0, 0
	real_TP, real_FP, real_TN, real_FN = 0, 0, 0, 0
	fake_TP, fake_FP, fake_TN, fake_FN = 0, 0, 0, 0
	
	start_time = time()
	with torch.no_grad():
		for step, (x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref) in enumerate(loader):
			'''eval here'''
			x, x_ref, y, y_ref,d, d_p = x.to(device), x_ref.to(device), y.to(device), y_ref.to(device),d.to(device), d_p.to(device)
			
			z = Encrypter(x)
			z_ref = Encrypter(x_ref)

			#segmentation
			y_hat = Segmentator(z)
			y_hat_ref = Segmentator(z_ref)
			seg_loss_1 = segmentation_criterion(y_hat, y)
			seg_loss_2 = segmentation_criterion(y_hat_ref, y_ref)
			seg_loss = seg_loss_1 + seg_loss_2
			run_seg_loss += seg_loss.item()
	
			pred_1 = torch.round(y_hat).detach()
			dice_score += compute_dice_score(pred_1, y)
			pred_2 = torch.round(y_hat_ref).detach()
			dice_score += compute_dice_score(pred_2, y_ref)
			
			#discriminator
			d_x_xref = Discriminator(x, x_ref)	
			d_x_z = Discriminator(x,z)
			d_x_zref = Discriminator(x, z_ref)
			d_xref_z = Discriminator(x_ref,z)
			d_xref_zref = Discriminator(x_ref, z_ref)
			d_z_zref = Discriminator(z, z_ref)
		
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
	dis_acc = (TP + TN) / (TP + FP + TN + FN)
	
	dis_real_acc = (real_TP + real_TN) / (real_TP + real_FP + real_TN + real_FN)
	dis_fake_acc = (fake_TP + fake_TN) / (fake_TP + fake_FP + fake_TN + fake_FN)
	
	print('|Validation: ----------------------------')
	print('Seg_loss:{:.4f} | Adv_loss:{:.4f} | Dis_loss:{:.4f}'.format(seg_loss, -adv_loss, dis_loss))
	print('dis_real_acc:{:.4f} | TP:{} FP:{} TN:{} FN:{} '.format(dis_real_acc, real_TP, real_FP, real_TN, real_FN))
	print('dis_fake_acc:{:.4f} | TP:{} FP:{} TN:{} FN:{} '.format(dis_fake_acc, fake_TP, fake_FP, fake_TN, fake_FN))
	print('duration:{:.0f}'.format(dur))
	
	return seg_loss, adv_loss, dis_loss,\
		   dis_real_acc, dis_fake_acc

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device')
	parser.add_argument('')
	train()

if __name__ == '__main__':
	main()

