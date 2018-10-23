# encoding= utf-8
from PIL import Image
import numpy as np
import os
import math
import cv2
import codecs
from config import WORDDICT, log


def sample_words():
    text_dir = '../data/train/txt_train'
    count = set(['', ])
    for txtname in os.listdir(text_dir):
        with open(text_dir + '/' + txtname, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                text = line.split(',')[-1].strip()
                if text == '###':
                    continue
                textSet = set(text)
                count.update(textSet)
                log.debug('textSet now: %s' % (textSet,))
    log.info('%s char found.' % (len(count),))

    # for txtname in os.listdir(text_dir):
    #     with open(os.path.join(text_dir, txtname), 'r', encoding="utf-8") as f:
    #         for line in f.readlines():
    #             text = line.split(',')[-1].strip()
    #             textSet = set(text)
    #             for c in textSet:
    #                 assert c in count

    for character in count:
        log.debug(character)
    with open(WORDDICT, 'w', encoding='utf-8') as f:
        for character in count:
            f.write(character)
            f.write('\n')



if __name__ == '__main__':
    sample_words()