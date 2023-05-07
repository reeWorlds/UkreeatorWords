# -*- coding: windows-1251 -*-

import re

text = ""

with open("../../../data/original/geo.ukr.txt", "r", encoding="utf-8") as f:
	text = f.read()

words_all = set(re.findall(r'[А-ЯЄЇІҐа-яєіїґ\'\-]+', text))
words = set()

for word in words_all:
	if len(word) > 2:
		words.add(word.lower())

print(len(words_all))
print(len(words))

def writeInFile(names, path):
	with open(path, 'w') as f:
		f.write("\n".join(names))

writeInFile(words, "../../../data/locations2.txt")