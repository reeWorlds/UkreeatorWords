# -*- coding: windows-1251 -*-

def readFromFile(path):
	with open(path, "r") as f:
		words = set(f.read().split("\n"))
	words.discard("")
	return words

def writeInFile(path, words):
	with open(path, "w") as f:
		f.write("\n".join(list(words)))



def fixLocations():
	path = "../../../data/locations.txt"
	words = readFromFile(path)
	words_new = set()
	for word in words:
		if "ы" not in word and "ъ" not in word:
			words_new.add(word)
	writeInFile(path, words_new)

#fixLocations()

def fixLocations2():
	pass

def fixNames2():
	pass

def fixNames3():
	path = "../../../data/names3.txt"
	words = readFromFile(path)
	words_new = set()
	for word in words:
		word = word.replace("c", "к")
		word = word.replace("q", "г")
		word = word.replace("w", "в")
		word = word.replace("x", "кс")
		words_new.add(word)
	writeInFile(path, words_new)

#fixNames3()

def fixWords1():
	path = "../../../data/words1.txt"
	words = readFromFile(path)
	words_new = set()
	for word in words:
		if "ы" not in word and "э" not in word:
			words_new.add(word)
	writeInFile(path, words_new)

#fixWords1()