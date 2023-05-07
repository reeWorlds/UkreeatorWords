#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <locale>
#include <codecvt>
using namespace std;

string path = "../../../data/original/uk.txt";

void createSmall()
{
	const int32_t len = 100001;

	FILE* fin = fopen(path.c_str(), "rb");

	uint8_t* data = new uint8_t[len];
	fread(data, sizeof(uint8_t), len, fin);

	fclose(fin);

	FILE* fout = fopen("part.txt", "wb");

	fwrite(data, sizeof(uint8_t), len, fout);

	fclose(fout);

	delete[] data;
}

set <wchar_t> getAlphabet()
{
	wifstream inFile("WordsChars.txt", std::ios::in);
	inFile.imbue(locale());

	set <wchar_t> alphabet;
	wchar_t c;

	while (inFile >> c)
	{
		alphabet.insert(c);
	}
	
	return alphabet;
}

bool isCapital(wchar_t c)
{
	if ((c >= L'A' && c <= L'ß') || c == L'¯' || c == L'²' || c == L'ª' || c == L'¥') { return true; }
	return false;
}

wstring lower(wstring word)
{
	for (int32_t i = 0; i < word.size(); i++)
	{
		if (isCapital(word[i]))
		{
			if (word[i] >= L'À' && word[i] <= L'ß')
			{
				word[i] = word[i] - L'À' + L'à';
			}
			else if (word[i] == L'¯') { word[i] = L'¿'; }
			else if (word[i] == L'²') { word[i] = L'³'; }
			else if (word[i] == L'ª') { word[i] = L'º'; }
			else if (word[i] == L'¥') { word[i] = L'´'; }
			else
			{
				cout << "Unprocessed symbol\n";
			}
		}
	}

	return word;
}

bool isOk(wstring word)
{
	if (word.size() < 3) { return 0; }

	if (word[0] == L'-' || word[0] == L'\'')
	{
		return false;
	}

	for (int32_t i = 1; i < word.size(); i++)
	{
		if ((word[i - 1] == L'\'' && word[i] == L'\'') ||
			(word[i - 1] == L'\'' && word[i] == L'-') || 
			(word[i - 1] == L'-' && word[i] == L'\'') || 
			(word[i - 1] == L'-' && word[i] == L'-'))
		{
			return false;
		}

		if (isCapital(word[i]) && word[i - 1] != L' ')
		{
			return false;
		}
	}

	return true;
}

void makeDict(string path, string outPath)
{
	locale::global(locale(locale(""), new codecvt_utf8<wchar_t>));

	wifstream in(path, ios::in);
	wstring line;

	in.imbue(locale());

	set <wchar_t> alphabet = getAlphabet();
	uint8_t* inAlp = new uint8_t[1 << (8 * sizeof(wchar_t))];
	memset(inAlp, 0, 1 << (8 * sizeof(wchar_t)));
	for (auto c : alphabet) { inAlp[c] = 1; }

	map <wstring, int32_t> words;

	int64_t lineI = 0;

	while(getline(in, line))
	{
		wstring word = L"";
	
		for (auto c : line)
		{
			if (alphabet.find(c) != alphabet.end())
			{
				word += c;
			}
			else
			{
				if (isOk(word))
				{
					words[lower(word)]++;
				}
				word = L"";
			}
		}
		if (isOk(word))
		{
			words[lower(word)]++;
		}

		lineI++;
		if (lineI % 100000 == 0)
		{
			cout << "lineI = " << lineI << " words.size() = " << words.size() << "\n";
		}
	}

	in.close();

	cout << "Total number of unique words is " << words.size() << "\n";

	wofstream out(outPath, ios::out);

	out.imbue(locale());

	vector <pair<int32_t, wstring> > wordsVec;
	for (auto word : words)
	{
		wordsVec.push_back(make_pair(word.second, word.first));
	}

	sort(wordsVec.begin(), wordsVec.end());
	reverse(wordsVec.begin(), wordsVec.end());

	wstring_convert<codecvt_utf8_utf16<wchar_t>> converter;

	for (auto word : wordsVec)
	{
		string num = to_string(word.first);
		wstring wnum = converter.from_bytes(num);

		out << word.second << " " << wnum << "\n";
	}

	out.close();
}

void makeDictSmall()
{
	makeDict("part.txt", "part.vocab.txt");
}

void makeDictBig()
{
	makeDict(path, "CntWord2List.txt");
}

void makeFinalDict(string pathIn, string pathOut, int32_t lim)
{
	locale::global(locale(locale(""), new codecvt_utf8<wchar_t>));

	wifstream in(pathIn, ios::in);
	in.imbue(locale());

	wofstream out(pathOut, ios::out);
	out.imbue(locale());

	vector <pair<int32_t, wstring> > wordsVec;

	wstring word;
	int32_t cnt;

	while (in >> word >> cnt)
	{
		if (cnt >= lim)
		{
			out << word << "\n";
		}
	}

	in.close();
	out.close();
}




int main()
{
	//createSmall();

	//makeDictSmall();

	//makeDictBig();

	//makeFinalDict("../../../data/original/CntWord2List.txt", "../../../data/words2_1.txt", 10);
	//makeFinalDict("../../../data/original/CntWord2List.txt", "../../../data/words2_2.txt", 0);

	return 0;
}