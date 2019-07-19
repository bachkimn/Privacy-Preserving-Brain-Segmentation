import os
import torch
import numpy as np
import nibabel as nib
from random import randint, choice
from torch.utils.data import Dataset, DataLoader
from ultils import *

class ppmi_pairs(Dataset):
	def __init__(self, data_folder, mode='train', ratio=0.5):
		self.mode = mode
		self.ratio = ratio
		self.data_folder = data_folder
		self.im_list = os.listdir(data_folder)
		self.positives, self.negatives = get_examples(self.im_list)
		self.patches = gen_crop_point()
		print('||- Loaded:"{}" as {} set'.format(self.data_folder, self.mode))
		print('| No of positives: {}, No of negatives:{}'.format(len(self.positives),len(self.negatives)))

	def __len__(self):
		if self.mode.lower() =='train':
			return int(len(self.positives)/self.ratio)
		else:
			return (len(self.positives)+len(self.negatives))
	
	def __get_pos_item__(self, index):
		label = 1
		point = choice(self.patches)
		im, im_ref = self.positives[index]
		
		x = load_nii_to_tensor(os.path.join(self.data_folder,im,'T1.nii.gz'), point)
		x_ref = load_nii_to_tensor(os.path.join(self.data_folder,im_ref,'T1.nii.gz'), point)
		
		y = load_segmap_to_tensor(os.path.join(self.data_folder,im,'segmap.nii.gz'), point)
		#y_ref = load_segmap_to_tensor(os.path.join(self.data_folder,im_ref,'segmap.nii.gz'), point)
		
		return	x, y, x_ref, label, im, im_ref
	
	def __get_neg_item__(self, index):
		label = 0
		point = choice(self.patches)
		if self.mode.lower() == 'train':
			im, im_ref = choice(self.negatives)
		else:
			im, im_ref = self.negatives[index]
			
		x = load_nii_to_tensor(os.path.join(self.data_folder,im,'T1.nii.gz'), point)
		x_ref = load_nii_to_tensor(os.path.join(self.data_folder,im_ref,'T1.nii.gz'), point)
		
		y = load_segmap_to_tensor(os.path.join(self.data_folder,im,'segmap.nii.gz'), point)
		#y_ref = load_segmap_to_tensor(os.path.join(self.data_folder,im_ref,'segmap.nii.gz'), point)
		
		return x, y, x_ref, label, im, im_ref
				
	def __getitem__(self, index):
		
		if index < len(self.positives):
			idx = index
			x, y, x_ref, label, im, im_ref = self.__get_pos_item__(idx)
			
		else:
			idx = index - len(self.positives)
			x, y, x_ref, label, im, im_ref = self.__get_neg_item__(idx)
		
		l_d1 = 1
		l_d2 = label
		l_ad1 = 0
		l_ad2 = 1 - l_d2
		#print(label,patch_no,idx,index,im1,im2)
		return x, x_ref, y, label, l_d1, l_d2, l_ad1, l_ad2, im, im_ref
