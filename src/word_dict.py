#coding:UTF-8
from config import  log, CHARFILE
def extract_charset():
	words= set()
	with open(CHARFILE, 'r', encoding="utf-8") as f:
		for line in f.readlines():
			log.debug('Char: %s' %(line, ))
			words.update(line)

	return words

if __name__ == '__main__':
	outchar_set()