# encoding= utf-8
from PIL import Image
import numpy as np
import os
import math
import cv2
import codecs
from config import CHARFILE, log

if __name__ == '__main__':
    text_dir = 'Data/Train/txt_train'
    count = set()
    for txtname in os.listdir(text_dir):
        with open(text_dir + '/' + txtname, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                text = line.split(',')[-1].strip()
                if text == '###':
                    continue
                count.update(text)
                print(text)
    print(len(count))

with open(CHARFILE, 'w') as f:
    for character in count:
        f.write(character)
        f.write('\n')
