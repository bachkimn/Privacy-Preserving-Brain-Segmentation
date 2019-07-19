import os
import torch
import numpy as np
from skimage.measure import compare_ssim as ssim
import torch.utils.data as data
import matplotlib.pyplot as plt

def plot_dist(dist1,dist2,dist3,dist4,dist5,dist6, name):
	s = min(dist1.min(),dist2.min(),dist3.min(),dist4.min(),dist5.min(),dist6.min())
	e = max(dist1.max(),dist2.max(),dist3.max(),dist4.max(),dist5.max(),dist6.max())
	bins = np.linspace(s,e,100)
	plt.hist(dist1,bins=bins, alpha = 0.5, density=True, color = 'r', label='Positive | mean:{:.6f} |std: {:.6f}'.format(dist1.mean(),dist1.std()))
	plt.hist(dist2,bins=bins, alpha = 0.5, density=True, color = 'c', label='Negative | mean:{:.6f} |std: {:.6f}'.format(dist2.mean(),dist2.std()))
	plt.hist(dist3,bins=bins, alpha = 0.5, density=True, color = 'b', label='Positive Enc | mean:{:.6f} |std: {:.6f}'.format(dist3.mean(),dist3.std()))
	plt.hist(dist4,bins=bins, alpha = 0.5, density=True, color = 'm', label='Negative Enc | mean:{:.6f} |std: {:.6f}'.format(dist4.mean(),dist4.std()))
	plt.hist(dist5,bins=bins, alpha = 0.5, density=True, color = 'y', label='Positive OvE | mean:{:.6f} |std: {:.6f}'.format(dist5.mean(),dist5.std()))
	plt.hist(dist6,bins=bins, alpha = 0.5, density=True, color = 'g', label='Negative OvE | mean:{:.6f} |std: {:.6f}'.format(dist6.mean(),dist6.std()))
	plt.legend(loc='upper right')
	plt.savefig(name)
	plt.clf()
	plt.close()
	return

def boxes_plot(dist1,dist2,dist3,dist4,dist5,dist6, name):
	plt.boxplot([dist1,dist2,dist3,dist4,dist5,dist6])
	plt.savefig(name)
	plt.clf()
	plt.close()
	return

def plot_mssim(models, data_set, device, save_dir, epoch):

	loader = data.DataLoader(data_set, batch_size = 1, num_workers = 16, pin_memory=True, shuffle = False)
	Encrypter = models['enc'].eval()

	save_dir_ms_ssim = os.path.join(save_dir, 'ms_ssim')
	if os.path.exists(save_dir_ms_ssim) == False:
		os.mkdir(save_dir_ms_ssim)
	plot_name = os.path.join(save_dir_ms_ssim, str(epoch) + '.png')
	
	save_dir_boxes = os.path.join(save_dir, 'box_plot')
	if os.path.exists(save_dir_boxes) == False:
		os.mkdir(save_dir_boxes)
	boxes_plot_name = os.path.join(save_dir_boxes, str(epoch) + '.png')

	with torch.no_grad():
		o_ssim_pos = []
		o_ssim_neg = []
		e_ssim_pos = []
		e_ssim_neg = []
		oe_ssim_pos = []
		oe_ssim_neg = []
		
		for step, (x, x_ref, y, y_ref,d, d_p, d_n, im, im_ref) in enumerate(loader):
			x, x_ref = x.to(device), x_ref.to(device)

			z = Encrypter(x)
			z_ref = Encrypter(x_ref)
			
			x = x[0,0,:,:,:].cpu().numpy().astype(float)
			x_ref = x_ref[0,0,:,:,:].cpu().numpy().astype(float)
			z = z[0,0,:,:,:].cpu().numpy().astype(float)
			z_ref = z_ref[0,0,:,:,:].cpu().numpy().astype(float)
			
			_o_ssim = ssim(x,x_ref)
			_e_ssim = ssim(z,z_ref)
			_oe_ssim = ssim(z, x_ref)
			#print(step,im, im_ref, d.item())
			if d.item() == 1:
				o_ssim_pos += [_o_ssim]
				e_ssim_pos += [_e_ssim]
				oe_ssim_pos += [_oe_ssim]
			else:
				o_ssim_neg += [_o_ssim]
				e_ssim_neg += [_e_ssim]
				oe_ssim_neg += [_oe_ssim]

	o_ssim_pos = np.array(o_ssim_pos)
	e_ssim_pos = np.array(e_ssim_pos)
	oe_ssim_pos = np.array(oe_ssim_pos)
	o_ssim_neg = np.array(o_ssim_neg)
	e_ssim_neg = np.array(e_ssim_neg)
	oe_ssim_neg = np.array(oe_ssim_neg)

	plot_dist(o_ssim_pos,o_ssim_neg,e_ssim_pos,e_ssim_neg,oe_ssim_pos,oe_ssim_neg, plot_name)
	boxes_plot(o_ssim_pos,o_ssim_neg,e_ssim_pos,e_ssim_neg,oe_ssim_pos,oe_ssim_neg, boxes_plot_name)
	return
