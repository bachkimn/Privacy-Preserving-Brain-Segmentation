import os
import numpy as np
import torch
import torch.utils.data as data
import argparse

from segmentation_loss import Dice_Loss
from segmentation import Unet3D as segnet
from autoencoder import Unet3D_encoder as encoder
from discriminator import discriminator
from ppmi import ppmi_pairs

from time import time
from ultils import *
from train import train_epoch
from val import val_epoch
from ms_ssim import plot_mssim
from vis import vis_image as vis

def main():
	'''Configuration'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type = int)
	parser.add_argument('--LAMBDA', type = int)
	parser.add_argument('--save_dir', type = str)
	parser.add_argument('--encoder', type = str, default = None)
	parser.add_argument('--dis', type = str, default = None)
	parser.add_argument('--seg', type = str, default = None)
	parser.add_argument('--vis', type = str, default = None)
	args = parser.parse_args()
	
	LAMBDA = args.LAMBDA
	batch_size = args.batch_size
	learning_rate = 1e-4
	num_epochs = 500
	
	device = torch.device('cuda')
	
	save_dir = args.save_dir
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)
	
	if args.vis != None:
		vis_image = args.vis + '/T1.nii.gz'
		vis_segmap = args.vis + '/segmap.nii.gz'

	enc_path = args.encoder
	dis_path = args.dis
	seg_path = args.seg
	
	'''Initialize dataset'''
	train_set = ppmi_pairs(mode = 'train')
	val_set = ppmi_pairs(mode = 'val')
	
	'''Initialize networks'''	
	# init model encrypter
	Encrypter = encoder(1,1,16).to(device)
	if enc_path != None:
		print('-> Loaded pre-trained:{}'.format(enc_path))
		Encrypter.load_state_dict(torch.load(enc_path, map_location=device))
	Encrypter.train()
	# init discriminator
	Discriminator = discriminator().to(device)
	if dis_path != None:
		print('-> Loaded pre-trained:{}'.format(dis_path))
		Discriminator.load_state_dict(torch.load(dis_path, map_location=device))
	Discriminator.train()
	# init segmentator
	Segmentator = segnet(1,6,32).to(device)
	if seg_path != None:
		print('-> Loaded pre-trained:{}'.format(seg_path))
		Segmentator.load_state_dict(torch.load(seg_path, map_location=device))
	Segmentator.train()
	#
	models = {'enc': Encrypter, 'seg': Segmentator, 'dis': Discriminator}

	'''Initialize optimizer'''
	# declare loss function
	Segment_criterion = Dice_Loss()
	Discrimination_criterion = torch.nn.CrossEntropyLoss()
	criterions = {'seg': Segment_criterion, 'dis': Discrimination_criterion}

	# init optimizer
	params_es = [{"params": Encrypter.parameters()},{"params": Segmentator.parameters()}]
	optimizer_es = torch.optim.Adam(params_es, lr = learning_rate)

	params_d = [{"params": Discriminator.parameters()}]
	optimizer_d = torch.optim.Adam(params_d, lr = learning_rate)
	
	optimizers = {'es': optimizer_es, 'dis': optimizer_d}
	
	# initialize tracking variables
	train_epochs = []
	val_epochs = []
	track_train_seg_loss = []
	track_train_adv_loss = []
	track_train_dis_loss = []
	track_val_seg_loss = []
	track_val_adv_loss = []
	track_val_dis_loss = []
	
	track_train_dice_score = []
	track_val_dice_score = []
	
	track_train_dis_acc = []
	track_val_dis_acc = []
	
	track_train_dis_real_acc = []
	track_train_dis_fake_acc = []
	track_val_dis_real_acc = []
	track_val_dis_fake_acc = []
	
	for epoch in range(num_epochs):
		print('|==========================\nEPOCH:{}'.format(epoch + 1))
		
		'''Trainnig'''
		seg_loss, adv_loss, dis_loss,\
		dis_real_acc, dis_fake_acc,\
		models, optimizers = train_epoch(models, optimizers, criterions, LAMBDA, train_set, batch_size,device)
		
		#logging training values
		train_epochs += [epoch+1]
		track_train_seg_loss += [seg_loss]
		track_train_adv_loss += [adv_loss]
		track_train_dis_loss += [dis_loss]
		#track_train_dice_score += [dice_score]
		#track_train_dis_acc += [dis_acc]
		track_train_dis_real_acc += [dis_real_acc]
		track_train_dis_fake_acc += [dis_fake_acc]
 
		'''Validation'''
		if (epoch + 1) % 1 == 0:
			seg_loss, adv_loss, dis_loss,\
			dis_real_acc, dis_fake_acc = val_epoch(models, criterions, val_set, batch_size, device)
			
			#logging validation values
			val_epochs += [epoch+1]
			track_val_seg_loss += [seg_loss]
			track_val_adv_loss += [adv_loss]
			track_val_dis_loss += [dis_loss]
			#track_val_dice_score += [dice_score]
			#track_val_dis_acc += [dis_acc]
			track_val_dis_real_acc += [dis_real_acc]
			track_val_dis_fake_acc += [dis_fake_acc]
		
			#Plot learning curves
			plot_curves(train_epochs, track_train_seg_loss, val_epochs, track_val_seg_loss, save_dir, 'seg_loss', epoch + 1)
			plot_curves(train_epochs, track_train_adv_loss, val_epochs, track_val_adv_loss, save_dir, 'adv_loss', epoch + 1)
			plot_curves(train_epochs, track_train_dis_loss, val_epochs, track_val_dis_loss, save_dir, 'dis_loss', epoch + 1)
			#plot_curves(train_epochs, track_train_dice_score, val_epochs, track_val_dice_score, save_dir, 'dice_score', epoch + 1)
			#plot_curves(train_epochs, track_train_dis_acc, val_epochs, track_val_dis_acc, save_dir, 'dis_acc', epoch + 1)
			plot_curves(train_epochs, track_train_dis_real_acc, val_epochs, track_val_dis_real_acc, save_dir, 'dis_real_acc', epoch + 1)
			plot_curves(train_epochs, track_train_dis_fake_acc, val_epochs, track_val_dis_fake_acc, save_dir, 'dis_fake_acc', epoch + 1)

			'''plot ms-ssim'''
			plot_mssim(models, val_set, device, save_dir, epoch + 1)
			if os.path.exists(vis_image) == True:
				vis(models, vis_image, vis_segmap, device, save_dir, epoch + 1)
		if (epoch + 1) % 25 == 0:
			'''save models'''
			models_dir = os.path.join(save_dir,'models')
			if os.path.exists(models_dir) == False:
				os.mkdir(models_dir)
			torch.save(models['enc'].state_dict(), os.path.join(models_dir, str(epoch + 1) + '_enc.pt'))
			torch.save(models['seg'].state_dict(), os.path.join(models_dir, str(epoch + 1) + '_seg.pt'))
			torch.save(models['dis'].state_dict(), os.path.join(models_dir, str(epoch + 1) + '_dis.pt'))
			torch.save(optimizers['es'].state_dict(), os.path.join(models_dir, str(epoch + 1) + '_optim_es.pt'))			
			torch.save(optimizers['dis'].state_dict(), os.path.join(models_dir, str(epoch + 1) + '_optim_dis.pt'))			

	return

if __name__ == '__main__':
	main()

