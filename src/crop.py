from PIL import Image
import numpy as np
import os
import math
import cv2
import codecs
from config import  log


def clock(coor):
    pos = coor.argsort(axis=1)
    coor = coor[:, pos[0]]
    if coor[1][0] > coor[1][1]:
        coor[:, [0, 1]] = coor[:, [1, 0]]
    if coor[1][2] < coor[1][3]:
        coor[:, [2, 3]] = coor[:, [3, 2]]
    # print(coor)
    # res = coor.tolist()

    return coor[0], coor[1]

def crop():
    root_dir = '../data/originData'
    text_dir = os.path.join(root_dir, 'txt_train')
    img_dir = os.path.join(root_dir, 'image_train')
    save_dir_hor = os.path.join(root_dir, 'crop_img_hor')
    save_dir_ver = os.path.join(root_dir, 'crop_img_ver')
    if not os.path.exists(save_dir_hor):
        os.mkdir(save_dir_hor)
    if not os.path.exists(save_dir_ver):
        os.mkdir(save_dir_ver)
    i = 0
    fh = codecs.open(save_dir_hor + '/label.txt', 'w', encoding='utf-8')
    fv = codecs.open(save_dir_ver + '/label.txt', 'w', encoding='utf-8')
    for imgfile in os.listdir(img_dir):
        i = i + 1
        img = Image.open(os.path.join(img_dir, imgfile)).convert('RGB')
        txtname = imgfile.split('.')[0:-1]
        imgname = '.'.join(txtname)
        txtname.append('txt')
        txtname = '.'.join(txtname)

        # For test:
        if i == 102:
            break

        with open(os.path.join(text_dir, txtname), 'r', encoding="utf-8") as f:
            for j, line in enumerate(f.readlines()):
                txt = line.split(',')
                coordinates = [float(i) for i in txt[0:8]]
                label = txt[-1].strip()
                if label == '###':
                    continue
                X = [coordinates[0], coordinates[2], coordinates[4], coordinates[6]]
                Y = [coordinates[1], coordinates[3], coordinates[5], coordinates[7]]
                X, Y = clock(np.array([X, Y]))
                w1 = math.sqrt((X[0] - X[2]) ** 2 + (Y[0] - Y[2]) ** 2)
                w2 = math.sqrt((X[1] - X[3]) ** 2 + (Y[1] - Y[3]) ** 2)
                h1 = math.sqrt((X[0] - X[1]) ** 2 + (Y[1] - Y[0]) ** 2)
                h2 = math.sqrt((X[2] - X[3]) ** 2 + (Y[2] - Y[3]) ** 2)
                w = max(w1, w2)
                h = max(h1, h2)
                Pts1 = np.float32(np.array([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]]))
                Pts2 = np.float32(np.array([[0, 0], [0, h], [w, h], [w, 0]]))
                img = np.array(img)
                M = cv2.getPerspectiveTransform(Pts1, Pts2)
                dst = cv2.warpPerspective(img, M, (int(w), int(h)))
                img_new = Image.fromarray(dst)
                # name = '/'+str(i)+'_'+str(j)+'.jpg'
                name = imgname + '_' + str(j) + '.jpg'
                if w > h: # 横的图片
                    # 对image_new做resize
                    p = img_new.size[1] / 31
                    if p < 1:
                        continue
                    new_height = 31
                    new_width = int(img_new.size[0] / p)
                    log.debug('horizonal image: with width %+3s and height %+3s' %(new_width, new_height))
                    image_resized = img_new.resize((new_width, new_height))

                    image_resized.save(os.path.join(save_dir_hor, name))
                    fh.write(name + ' ' + label + '\n')
                else: # 竖的图片
                    p = img_new.size[0] / 31
                    if p < 1:
                        continue
                    new_height = int(img_new.size[1] / p)
                    new_width = 31
                    log.debug('vertical image: with width %+3s and height %+3s' %(new_width, new_height))
                    image_resized = img_new.resize((new_width, new_height))

                    image_resized.save(os.path.join(save_dir_ver, name))
                    fv.write(name + ' ' + label + '\n')
    # break
    fh.close()
    fv.close()

if __name__ == '__main__':
    crop()