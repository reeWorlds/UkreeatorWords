import torch
import torch.nn as nn
import random
import torch.nn.functional as F

def readWords(path):
	words = set()
	try:
		with open(path, 'r', encoding="utf-8") as f:
			words.update(set(f.read().split("\n")))
	except:
		with open(path, 'r', encoding="cp1251") as f:
			words.update(set(f.read().split("\n")))
	words.discard("")
	return words

def getAllWords(listNames):
	allWords = set()

	for name in listNames:
		allWords.update(readWords("../../../data/" + name + ".txt"))

	return list(allWords)

words_set = getAllWords(["locations", "locations2", "names2", "names3", "words1", "words2_1", "words2_2"])
print(f"full words list size = {len(words_set)}")

chars = set()
for word in words_set:
	chars.update(set(word))
chars = sorted(list(chars))

print("Number of unique chars = " + str(len(chars)))
print("".join(chars))

ctoi = dict([(x, i) for i, x in enumerate(chars)])
itoc = dict([(i, x) for i, x in enumerate(chars)])
delim = len(chars)
size = delim + 1
ctoi['.'] = delim
itoc[delim] = '.'


input_len = 11

def align_word(word):
	len_diff = (input_len - 1) - len(word)

	if len_diff < 0:
		word = word[-input_len + 1:]
	if len_diff > 0:
		word = ("." * len_diff) + word
	word = word + "."

	return word

def split_word_into_tokens(word):
	res = []
	for i in range(len(word)):
		res.append((align_word(word[0:i]), word[i]))
	res.append((align_word(word), "."))

	return res

def str_to_int(samples):
	return [([ctoi[c] for c in sample[0]], ctoi[sample[1]]) for sample in samples]

for x,y in split_word_into_tokens(words_set[0]):
	print(f"{x} -> {y}")

all_samples = []
for word in words_set:
	all_samples.extend(str_to_int(split_word_into_tokens(word)))
random.shuffle(all_samples)

print(f"total of {len(all_samples)} tokens")

n_split = int(len(all_samples) * 0.9)

x_trn = torch.tensor([sample[0] for sample in all_samples[0:n_split]])
y_trn = torch.tensor([sample[1] for sample in all_samples[0:n_split]])
x_vld = torch.tensor([sample[0] for sample in all_samples[n_split:]])
y_vld = torch.tensor([sample[1] for sample in all_samples[n_split:]])

print(f"train shape x = {x_trn.shape} y = {y_trn.shape}")
print(f"validation shape x = {x_vld.shape} y = {y_vld.shape}")

batch_size = 256
steps_in_epoch = n_split // batch_size

class Model(nn.Module):
	def __init__(self, n_embd, l1_n, l2_n, l3_n, l4_n, head_n, leaky_relu_a):
		super(Model, self).__init__()
		
		self.embedding = nn.Embedding(size, n_embd)
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
		
		self.head = nn.Linear(head_n, size)

	def forward(self, x):
		x = self.embedding(x).permute(0, 2, 1)
		x = self.activ1(self.batch1(self.l1(x)))
		x = self.activ2(self.batch2(self.l2(x)))
		x = self.activ3(self.batch3(self.l3(x)))
		x = self.activ4(self.batch4(self.l4(x)))
		x = self.activ5(self.batch5(self.l5(self.flat(x))))
		x = self.head(x)

		return x


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def getScore(model, device):
	model.eval()
	
	test_batch = 2048
	train_llh, valid_llh = 0.0, 0.0
	train_tot, valid_tot = 0, 0

	with torch.no_grad():
		for i in range(test_batch, x_vld.size(0), test_batch):
			idx = torch.arange(i - test_batch, i, 1)
			x, y = x_vld[idx].to(device), y_vld[idx].to(device)
			
			outputs = model(x)
			llh = F.cross_entropy(outputs, y)
			
			valid_llh += llh.item() * test_batch
			valid_tot += test_batch
  
	with torch.no_grad():
		for i in range(test_batch, x_trn.size(0), test_batch):
			idx = torch.arange(i - test_batch, i, 1)
			x, y = x_trn[idx].to(device), y_trn[idx].to(device)
			
			outputs = model(x)
			llh = F.cross_entropy(outputs, y)
			
			train_llh += llh.item() * test_batch
			train_tot += test_batch
  
	valid_llh /= valid_tot
	train_llh /= train_tot

	return train_llh, valid_llh


def pretrainModel(pathOut, n_embd, l1_n, l2_n, l3_n, l4_n, head_n, leaky_relu_a):
	print(f"Start training {pathOut}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Model(n_embd, l1_n, l2_n, l3_n, l4_n, head_n, leaky_relu_a)
	model.to(device)

	torch.manual_seed(74)

	optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 0.001)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

	print(f"pretraining model with {count_parameters(model)} parameters")

	train_llh, valid_llh = getScore(model, device)
	print(f"epoch 0 loss = {train_llh}   {valid_llh}")

	for epoch in range(10):
		model.train()
		
		for step in range(steps_in_epoch):
			optimizer.zero_grad()
			
			ix = torch.randint(0, x_trn.size(0), (batch_size,))
			x, y = x_trn[ix].to(device), y_trn[ix].to(device)
			
			outputs = model(x)
			loss = F.cross_entropy(outputs, y)
			
			loss.backward()
			optimizer.step()
  
		scheduler.step()

		for param_group in optimizer.param_groups:
			param_group['weight_decay'] = [param_group['lr'] for param_group in optimizer.param_groups][0] * 0.1


		train_llh, valid_llh = getScore(model, device)
		print(f"epoch {epoch + 1} loss = {train_llh}   {valid_llh}")

	torch.save(model.state_dict(), pathOut)


if __name__ == "__main__":
	#pretrainModel("../../../models/pretrained/small.pth", 2, 4, 8, 8, 12, 12, 0.05)
	#pretrainModel("../../../models/pretrained/normal.pth", 16, 16, 24, 32, 48, 64, 0.05)
	#pretrainModel("../../../models/pretrained/extra.pth", 32, 48, 80, 96, 128, 128, 0.05)

	pass