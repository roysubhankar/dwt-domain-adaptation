import torch
import torch.nn as nn
import torch.nn.functional as F

class MinEntropyConsensusLoss(nn.Module):
	def __init__(self, num_classes, device):
		super(MinEntropyConsensusLoss, self).__init__()
		self.num_classes = num_classes
		self.device = device

	def forward(self, x, y):
		i = torch.eye(self.num_classes, device=self.device).unsqueeze(0)
		x = F.log_softmax(x, dim=1)
		y = F.log_softmax(y, dim=1)

		x = x.unsqueeze(-1)
		y = y.unsqueeze(-1)

		ce_x = (- 1.0 * i * x).sum(1)
		ce_y = (- 1.0 * i * y).sum(1)

		ce = 0.5 * (ce_x + ce_y).min(1)[0].mean()

		return ce




