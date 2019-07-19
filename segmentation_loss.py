import torch
import torch.nn as nn

class Dice_Loss(nn.Module):
	def __init__(self):
		super(Dice_Loss, self).__init__()
	
	def forward(self, pred, target):
		'''things to do: exclude background, separate loss for each class'''
		smooth = 1.
		intersection = (pred * target).sum()
		union = (pred + target).sum()
		loss = 1 - (2 * intersection + smooth)/(union + smooth)
		return loss
