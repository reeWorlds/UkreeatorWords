# -*- coding: windows-1251 -*-

from transliterate import translit
import re

text = ""

with open("../../../data/original/names_eng.txt", "r", encoding="utf-8") as f:
	text = f.read()

names = set(re.findall(r'[a-z]+', text))

print(len(names))

def writeInFile(names, path):
	with open(path, 'w') as f:
		f.write("\n".join(names))

writeInFile([translit(name.replace("ph", "f"), 'uk', reversed=False) for name in names], "../../../data/names3.txt")
# translit(name, 'uk', reversed=True)