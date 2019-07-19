import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.utils.data as data
from ultils import *

def save_to_nii(im, name):
	im = nib.Nifti1Image(im, np.eye(4))
	nib.save(im, name + '.nii.gz')

#def load_nii_to_tensor(filename):
#	im = nib.load(filename).get_fdata()
#	im = (im - im.min()) / (im.max() - im.min())
#	im_tensor = torch.from_numpy(im).float().view(1, 1, im.shape[0], im.shape[1], im.shape[2])
#	return im_tensor

def plot(slices, name):
	fig = plt.figure()
	for i,im in enumerate(slices):
		fig.add_subplot(len(slices)/2,2,i+1)
		plt.imshow(im,cmap='gray')
	plt.savefig(name)
	plt.clf()
	return
	
def vis_image(models, image, segmap, device, save_dir, epoch):

	Encrypter = models['enc'].eval()
	Segmentator = models['seg'].eval()

	image_numpy = load_nii_to_numpy(image)
	segmap_numpy = load_nii_to_numpy(segmap)
	image_tensor = load_nii_to_tensor(image).to(device).view(1, 1, 144, 192, 160)
	
	with torch.no_grad():
		z = Encrypter(image_tensor)
		z = z.detach()
		del Encrypter
		torch.cuda.empty_cache()	
		segmap = Segmentator(z)
		segmap = segmap.argmax(dim=1)
		
		encoded_image = z[0,0,:,:,:].cpu().numpy()
		segmap = segmap[0,:,:,:].cpu().numpy().astype(float)
	slice_1 = image_numpy[:, :, 80]
	slice_2 = segmap_numpy[:, :, 80]
	slice_3 = encoded_image[:, :, 80]
	slice_4 = segmap[:, :, 80]
	slices = [slice_1, slice_2, slice_3, slice_4]
	save_dir = os.path.join(save_dir, 'vis')
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)
	name = os.path.join(save_dir, str(epoch) + '.png')
	plot(slices, name)
	return




