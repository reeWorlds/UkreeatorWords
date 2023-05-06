# -*- coding: windows-1251 -*-

from xml.dom.pulldom import PROCESSING_INSTRUCTION
import xml.etree.ElementTree as ET
import re

tree = ET.parse('../../../data/original/26-ex_xml_atu.xml')
root = tree.getroot()

def find_all_tags():
	def collect_all_tags():
		def get_all_tags(element, tags):
			for child in element:
				tags.add(child.tag)
				get_all_tags(child, tags)

		tags = set()
		get_all_tags(root, tags)

		with open("tags.txt", "w") as f:
			for tag in tags:
				f.write(tag + '\n')

		return tags

	# tags = collect_all_tags()
	tags = set()
	with open("tags.txt", "r") as f:
		for line in f:
			tags.add(line.strip())


def get_all_elements_with_tag(tag):
	elements = root.findall(".//" + tag)
	
	all_elements = set()
	for element in elements:
		all_elements.add(element.text)

	return all_elements

def writeInFile(names, path):
	with open(path, 'w') as f:
		f.write("\n".join(names))

def readFromFile(path):
	res = set()
	with open(path, 'r') as f:
		res = set(f.read().split("\n"))

	res.discard('')

	return res

def process_city_name():
	all_city = get_all_elements_with_tag("CITY_NAME")

	res1 = set()
	res2_1, res2_2 = set(), set()
	res3 = set()
	res4 = set()
	res5 = set()
	res6 = set()
	res7 = set()
	res8 = set()
	unvisited = set()

	for city in all_city:
		if city is None:
			continue

		if bool(re.match(r'^с.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^с.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res1.add(match.group(1))
		elif bool(re.match(r'^с.[А-ЯЄІЇҐа-яєіїґ \-\']+, [А-ЯЄІЇҐа-яєіїґ \-\']+ ТГ$', city)):
			match = re.search(r'^с.([А-ЯЄІЇҐа-яєіїґ \-\']+), ([А-ЯЄІЇҐа-яєіїґ \-\']+) ТГ$', city)
			res2_1.add(match.group(1))
			res2_2.add(match.group(2))
		elif bool(re.match(r'^м.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^м.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res3.add(match.group(1))
		elif bool(re.match(r'^с/рада.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^с/рада.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res4.add(match.group(1))
		elif bool(re.match(r'^сщ.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^сщ.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res5.add(match.group(1))
		elif bool(re.match(r'^смт.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^смт.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res6.add(match.group(1))
		elif bool(re.match(r'^сщ/рада.[А-ЯЄІЇҐа-яєіїґ \-\']+$', city)):
			match = re.search(r'^сщ/рада.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', city)
			res7.add(match.group(1))
		else:
			words = re.findall(r'[А-ЯЄЇІҐ][а-яєіїґ\']+', city)
			if len(words) > 0:
				for word in words:
					res8.add(word)
			else:
				unvisited.add(city)
	
	full = res1 | res2_1 | res2_2 | res3 | res4 | res5 | res6 | res7 | res8

	print("res1 size = " + str(len(res1)))
	print("res2_1 size = " + str(len(res2_1)) + "  res2_2 size = " + str(len(res2_2)))
	print("res3 size = " + str(len(res3)))
	print("res4 size = " + str(len(res4)))
	print("res5 size = " + str(len(res5)))
	print("res6 size = " + str(len(res6)))
	print("res7 size = " + str(len(res7)))
	print("res8 size = " + str(len(res8)))
	print(unvisited)

	print("full size = " + str(len(full)))

	writeInFile(res1, "city/res1.txt")
	writeInFile(res2_1, "city/res2_1.txt")
	writeInFile(res2_2, "city/res2_2.txt")
	writeInFile(res3, "city/res3.txt")
	writeInFile(res4, "city/res4.txt")
	writeInFile(res5, "city/res5.txt")
	writeInFile(res6, "city/res6.txt")
	writeInFile(res7, "city/res7.txt")
	writeInFile(res8, "city/res8.txt")

	writeInFile(full, "city.txt")


def process_street_name():
	all_street = list(get_all_elements_with_tag("STREET_NAME"))

	res1, res1_2 = set(), set()
	res2 = set()
	res3 = set()
	unvisited = set()

	for street in all_street:
		if street is None:
			continue

		if bool(re.match(r'^вул.[А-ЯЄІЇҐа-яєіїґ \-\']+$', street)):
			match = re.search(r'^вул.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', street)
			res1.add(match.group(1))
		elif bool(re.match(r'^вул.[А-ЯЄІЇҐа-яєіїґ \-\']+ [А-ЯЄЇІҐ].$', street)):
			match = re.search(r'^вул.([А-ЯЄІЇҐа-яєіїґ \-\']+) [А-ЯЄЇІҐ].$', street)
			res1_2.add(match.group(1))
		elif  bool(re.match(r'^пров.[А-ЯЄІЇҐа-яєіїґ \-\']+$', street)):
			match = re.search(r'^пров.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', street)
			res2.add(match.group(1))
		else:
			words = re.findall(r'[А-ЯЄЇІҐ][а-яєіїґ\']+', street)
			if len(words) > 0:
				for word in words:
					res3.add(word)
			elif len(re.findall(r'[0-9]', street)):
				pass
			else:
				unvisited.add(street)
	
	for s in ['П\'ятанчука', 'П\'ятихатки', 'П\'ятака', 'П\'ятихатська', 'П\'ятихатський',\
	   'П\'яскорського', 'Мічурівський', 'П\'ятигірський', 'П\'ята', 'В\'язівненський', 'В\'яземський']:
		res3.add(s)

	full = res1 | res1_2 | res2 | res3

	print("res1 size = " + str(len(res1)))
	print("res1_2 size = " + str(len(res1_2)))
	print("res2 size = " + str(len(res2)))
	print("res3 size = " + str(len(res3)))
	print(unvisited)
	
	print("full size = " + str(len(full)))
	
	writeInFile(res1, "street/res1.txt")
	writeInFile(res1_2, "street/res1_2.txt")
	writeInFile(res2, "street/res2.txt")
	writeInFile(res3, "street/res3.txt")
	
	writeInFile(full, "street.txt")


def process_obl_name():
	all_obl = list(get_all_elements_with_tag("OBL_NAME"))

	full = set()
	rest = set()

	for obl in all_obl:
		if bool(re.match(r'^[А-ЯЄІЇҐа-яєіїґ \-\']+ обл.$', obl)):
			match = re.search(r'^([А-ЯЄІЇҐа-яєіїґ \-\']+) обл.$', obl)
			full.add(match.group(1))
		else:
			rest.add(obl)

	full.update({'Крим', 'Севастополь', 'Київ'})

	print(full)
	#print(rest)

	writeInFile(full, "obl.txt")


def process_region_name():
	all_region = list(get_all_elements_with_tag("REGION_NAME"))

	full = set()
	rest = set()

	for region in all_region:
		if region is None:
			continue

		if bool(re.match(r'^р.Райони [А-ЯЄІЇҐа-яєіїґ \-\']+$', region)):
			pass
		elif bool(re.match(r'^Райони [А-ЯЄІЇҐа-яєіїґ \-\']+$', region)):
			pass
		elif bool(re.match(r'^[А-ЯЄІЇҐа-яєіїґ \-\']+ р-н$', region)):
			match = re.search(r'^([А-ЯЄІЇҐа-яєіїґ \-\']+) р-н$', region)
			full.add(match.group(1))
		elif bool(re.match(r'^р.[А-ЯЄІЇҐа-яєіїґ \-\']+$', region)):
			match = re.search(r'^р.([А-ЯЄІЇҐа-яєіїґ \-\']+)$', region)
			full.add(match.group(1))
		else:
			rest.add(region)

	print("full size = " + str(len(full)))
	print(rest)

	writeInFile(full, "region.txt")


def process_city_region_name():
	all_city_region = list(get_all_elements_with_tag("CITY_REGION_NAME"))

	full = set()
	rest = set()

	for item in all_city_region:
		if item is None:
			continue

		words = re.findall(r'[А-ЯЄЇІҐ][а-яєіїґ\']+', item)
		if len(words) > 0:
			for word in words:
				full.add(word)
		else:
			rest.add(item)

	print("full size = " + str(len(full)))
	print(rest)

	writeInFile(full, "city_region.txt")


def merge_all_files():
	obl = readFromFile("obl.txt")
	region = readFromFile("region.txt")
	street = readFromFile("street.txt")
	city = readFromFile("city.txt")
	city_region = readFromFile("city_region.txt")
	 
	full = obl | region | street | city | city_region
	
	full_filtered = set()
	discarded= set()

	for name in full:
		words = re.split(r'[ \-]', name)
		isOk = len(name) > 2
		
		for word in words:
			if not bool(re.match('^[А-ЯЄІЇҐ][а-яєіїґ\']+$', word)) and not bool(re.match('^[а-яєіїґ\']+$', word)):
				isOk = False
		
		if '\'\'' in name:
			isOk = False

		if isOk:
			full_filtered.add(name.lower())
		else:
			discarded.add(name)

	separate_size = len(obl) + len(region) + len(street) + len(city) + len(city_region)
	
	print("separate size = " + str(separate_size))
	print("full size = " + str(len(full)))
	print("full_filtered size = " + str(len(full_filtered)))
	
	writeInFile(full_filtered, "../../../data/locations.txt")


if __name__ == "__main__":
	# find_all_tags()
	# RECORD, CITY_NAME, STREET_NAME, OBL_NAME, REGION_NAME, CITY_REGION_NAME
	
	#process_city_name()
	#process_street_name()
	#process_obl_name()
	#process_region_name()
	#process_city_region_name()

	#merge_all_files()

	pass