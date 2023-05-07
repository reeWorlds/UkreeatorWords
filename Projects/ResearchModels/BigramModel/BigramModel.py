# -*- coding: windows-1251 -*-

import random

def readWords(path):
	words = set()
	try:
		with open(path, 'r', encoding="utf-8") as f:
			words.update(set(f.read().split("\n")))
	except:
		with open(path, 'r') as f:
			words.update(set(f.read().split("\n")))
	words.discard("")
	return words

def getAllWords():
	allWords = set()

	for name in ["locations", "locations2", "names2", "names3", "words1", "words2_1", "words2_2"]:
	#for name in ["locations", "locations2", "names2", "names3", "words1"]:
		allWords.update(readWords("../../../data/" + name + ".txt"))

	return allWords

words = list(getAllWords())

chars = set()
for word in words:
	chars.update(set(word))
chars = sorted(list(chars))

print("Number of unique words = " + str(len(words)))
print("Number of unique chars = " + str(len(chars)))
print("".join(chars))

ctoi = dict([(x, i) for i, x in enumerate(chars)])
itoc = dict([(i, x) for i, x in enumerate(chars)])
delim = len(chars)
size = delim + 1
ctoi['#'] = delim
itoc[delim] = '#'

w = [[0] * size for i in range(size)]

print(ctoi)

for word in words:
	w[delim][ctoi[word[0]]] += 1
	for i in range(1, len(word)):
		w[ctoi[word[i - 1]]][ctoi[word[i]]] += 1
	w[ctoi[word[-1]]][delim] += 1

for i, row in enumerate(w):
	totCnt = sum(row)
	for j in range(len(row)):
		row[j] = row[j] / totCnt
	print(f"i = {i}  {itoc[i]} -> {itoc[row.index(max(row))]} with probability {max(row)}")

random.seed(42)

def generateWord():
	def getIndex(row, val):
		for i in range(len(row)):
			val -= row[i]
			if val <= 0.0:
				return i
		return len(row) - 1

	word = []
	last = delim
	while True:
		last = getIndex(w[last], random.uniform(0.0, 1.0))
		if last == delim:
			break
		word.append(itoc[last])

	return "".join(word)

for i in range(100):
	print(generateWord())