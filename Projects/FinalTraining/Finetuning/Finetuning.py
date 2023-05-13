# -*- coding: windows-1251 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


ctoi = {c : i for i, c in enumerate(" '-абвгдежзийклмнопрстуфхцчшщьюяєіїґ.")}

def read_words_file(path):
	words = {}

	try:
		with open(path, 'r', encoding="utf-8") as f:
			words = set(f.read().split("\n"))
	except:
		with open(path, 'r', encoding="cp1251") as f:
			words = set(f.read().split("\n"))

	words.discard("")

	return list(words)

def allign_word(word):
	len_diff = 10 - len(word)
	if len_diff < 0:
		return word[-10:] + "."
	elif len_diff > 0:
		return ("." * len_diff) + word + "."
	else:
		return word + "."

def load_and_prepare_dataet(name):
	words = read_words_file("../../../data/" + name)

	inputs, outputs = [], []

	for word in words:
		for i in range(len(word)):
			inputs.append([ctoi[c] for c in allign_word(word[0:i])])
			outputs.append(ctoi[word[i]])
		inputs.append([ctoi[c] for c in allign_word(word)])
		outputs.append(ctoi['.'])

	x = torch.tensor(inputs)
	y = torch.tensor(outputs)

	return x, y


class Model(nn.Module):
	def __init__(self, n_embd, l1_n, l2_n, l3_n, l4_n, head_n, leaky_relu_a):
		super(Model, self).__init__()
		
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

def load_pretrained_model(model_size):
	model = None

	if model_size == "tiny":
		model = Model(2, 4, 8, 8, 12, 12, 0.05)
		model.load_state_dict(torch.load("../../../models/pretrained/tiny.pth"))
	elif model_size == "normal":
		model = Model(16, 16, 24, 32, 48, 64, 0.05)
		model.load_state_dict(torch.load("../../../models/pretrained/normal.pth"))
	elif model_size == "extra":
		model = Model(32, 48, 80, 96, 128, 128, 0.05)
		model.load_state_dict(torch.load("../../../models/pretrained/extra.pth"))

	return model


def sampleW(prbs):
	w = torch.rand(1).item()
	for i in range(len(prbs)):
		w -= prbs[i]
		if w <= 0:
			return i
	return len(prbs) - 1

def finetune(names, prbs, model_size, steps_in_epoch, data_type):
	print(f"Start fine-tuning {model_size} model on {names}")

	dataX = []
	dataY = []

	for name in names:
		x, y = load_and_prepare_dataet(name)
		dataX.append(x)
		dataY.append(y)

	model = load_pretrained_model(model_size)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	batch_size = 128

	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
	
	for epoch in range(10):
		model.train()

		loss_sum, loss_cnt = 0.0, 0

		for step in range(steps_in_epoch):
			optimizer.zero_grad()

			wi = sampleW(prbs)
			ix = torch.randint(0, dataX[wi].size(0), (batch_size,))
			x, y = dataX[wi][ix].to(device), dataY[wi][ix].to(device)

			outputs = model(x)
			loss = F.cross_entropy(outputs, y)

			loss.backward()
			optimizer.step()

			loss_sum += loss.item()
			loss_cnt += 1

		print(f"epoch {epoch + 1} loss = {loss_sum / loss_cnt}")

		scheduler.step()
		for pg in optimizer.param_groups:
			pg['weight_decay'] = [pg['lr'] for pg in optimizer.param_groups][0] * 0.1

	torch.save(model.state_dict(), "../../../models/" + data_type + "/" + model_size + ".pth")


if __name__ == "__main__":
	#for model_size in ["tiny", "normal", "extra"]:
	#	finetune(["names2.txt", "names3.txt"], [0.8, 0.2], model_size, 1000, "names")
	#
	#print("\n\n\n")
	#
	#for model_size in ["tiny", "normal", "extra"]:
	#	finetune(["locations.txt", "locations2.txt"], [0.8, 0.2], model_size, 1000, "locations")
	#
	#print("\n\n\n")
	#
	#for model_size in ["tiny", "normal", "extra"]:
	#	finetune(["words1.txt", "words2_1.txt"], [0.6, 0.4], model_size, 1000, "words")

	pass