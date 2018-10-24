# coding:UTF-8
from config import log, WORDDICT
import os

def load_dict():
    # words = []
    # with open(WORDDICT, 'r', encoding="utf-8") as f:
    #     for line in f.readlines():
    #         log.debug('Char: %s' % (line.strip(),))
    #         words.append(line)
    # log.info('Count %s characters!' %(len(words), ))
    text_dir = '../data/originData/txt_train'
    words = set()
    for txtname in os.listdir(text_dir):
        with open(text_dir + '/' + txtname, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                text = line.split(',')[-1].strip()
                if text == '###':
                    continue
                textSet = set(text)
                words.update(textSet)
                # log.debug('textSet now: %s' % (textSet,))
    log.info('%s char found.' % (len(words),))

    return words


if __name__ == '__main__':
    words= load_dict()
    assert '1' in words
