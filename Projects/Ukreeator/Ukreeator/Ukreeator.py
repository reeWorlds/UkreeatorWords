# -*- coding: windows-1251 -*-

import torch
import torch.nn as nn

class WordsModel(nn.Module):
	def __init__(self, n_embd, l1_n, l2_n, l3_n, l4_n, head_n, leaky_relu_a):
		super(WordsModel, self).__init__()
		
		self.embedding = nn.Embedding(37, n_embd)
		nn.init.kaiming_normal_(self.embedding.weight, mode='fan_in')
		
		self.l1 = nn.Conv1d(n_embd, l1_n, 2)
		nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='leaky_relu', a=leaky_relu_a)
		self.batch1 = nn.BatchNorm1d(l1_n)
		self.activ1 = nn.LeakyReLU(leaky_relu_a)
		
		self.l2 = nn.Conv1d(l1_n, l2_n, 2, stride=2)
		nn.init.xavier_uniform_(self.l2.weight)
		self.batch2 = nn.BatchNorm1d(l2_n)
		self.activ2 = nn.Tanh()
		
		self.l3 = nn.Conv1d(l2_n, l3_n, 2)
		nn.init.xavier_uniform_(self.l3.weight)
		self.batch3 = nn.BatchNorm1d(l3_n)
		self.activ3 = nn.Tanh()
		
		self.l4 = nn.Conv1d(l3_n, l4_n, 2, stride=2)
		nn.init.xavier_uniform_(self.l4.weight)
		self.batch4 = nn.BatchNorm1d(l4_n)
		self.activ4 = nn.Tanh()
		
		self.flat = nn.Flatten()
		self.l5 = nn.Linear(2 * l4_n, head_n)
		nn.init.xavier_uniform_(self.l5.weight)
		self.batch5 = nn.BatchNorm1d(head_n)
		self.activ5 = nn.Tanh()
		
		self.head = nn.Linear(head_n, 37)

	def forward(self, x):
		x = self.embedding(x).permute(0, 2, 1)
		x = self.activ1(self.batch1(self.l1(x)))
		x = self.activ2(self.batch2(self.l2(x)))
		x = self.activ3(self.batch3(self.l3(x)))
		x = self.activ4(self.batch4(self.l4(x)))
		x = self.activ5(self.batch5(self.l5(self.flat(x))))
		x = self.head(x)

		return x


class UkreeatorWords():
	def __init__(self, model_size = None, weights_path = None):
		self.ctoi = {c : i for i, c in enumerate(" '-абвгдежзийклмнопрстуфхцчшщьюяєіїґ.")}
		self.itoc = {i : c for (c, i) in self.ctoi.items()}

		self.model = None
		if model_size is not None and weights_path is not None:
			self.load_model(model_size, weights_path)

	def load_model(self, model_size, weights_path):
		if model_size == "tiny":
			self.model = WordsModel(2, 4, 8, 8, 12, 12, 0.05)
			self.model.load_state_dict(torch.load(weights_path))
			self.model.eval()
		elif model_size == "normal":
			self.model = WordsModel(16, 16, 24, 32, 48, 64, 0.05)
			self.model.load_state_dict(torch.load(weights_path))
			self.model.eval()
		elif model_size == "extra":
			self.model = WordsModel(32, 48, 80, 96, 128, 128, 0.05)
			self.model.load_state_dict(torch.load(weights_path))
			self.model.eval()

	def align_word(self, word):
		len_diff = 10 - len(word)
		if len_diff < 0:
			return word[-10:] + "."
		elif len_diff > 0:
			return ("." * len_diff) + word + "."
		else:
			return word + "."

	def generateWord(self, prefix = ""):
		with torch.no_grad():
			while True:
				prefix_aligned = self.align_word(prefix)
				
				prefix_tensor = torch.tensor([[self.ctoi[c] for c in prefix_aligned]], dtype=torch.long)
				prefix_tensor.view(1, -1)
				
				prbs = torch.nn.functional.softmax(self.model.forward(prefix_tensor).view(-1), dim=0)
				letterI = torch.multinomial(prbs, 1).item()

				newC = self.itoc[letterI]
				if newC == ".":
					return prefix
				else:
					prefix += newC

	def set_torch_seed(self, seed):
		torch.manual_seed(seed)
