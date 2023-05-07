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
		if "�" not in word and "�" not in word:
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
		word = word.replace("c", "�")
		word = word.replace("q", "�")
		word = word.replace("w", "�")
		word = word.replace("x", "��")
		words_new.add(word)
	writeInFile(path, words_new)

#fixNames3()

def fixWords1():
	path = "../../../data/words1.txt"
	words = readFromFile(path)
	words_new = set()
	for word in words:
		if "�" not in word and "�" not in word:
			words_new.add(word)
	writeInFile(path, words_new)

#fixWords1()