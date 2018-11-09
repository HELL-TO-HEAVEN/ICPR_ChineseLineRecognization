from PIL import Image
import numpy as np
import os
import math
import cv2
from sklearn.model_selection import ShuffleSplit
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

def preprocess():
    root_dir = '../data/originData'
    text_dir = os.path.join(root_dir, 'txt_train')
    img_dir = os.path.join(root_dir, 'image_train')
    shuffleSplit = ShuffleSplit(n_splits=1, test_size=0.1)
    imgList = os.listdir(img_dir)
    for trainInd, valInd in shuffleSplit.split(imgList):
        trainDir= '../data/train'
        valDir= '../data/val'
        log.info('Train image crop Start:')
        crop([imgList[i] for i in trainInd], img_dir, text_dir, trainDir)
        log.info('Validation image crop Start:')
        crop([imgList[i] for i in valInd], img_dir, text_dir, valDir)

def crop(imgFiles, imgDir, txtDir, saveDir):
    save_dir_hor = os.path.join(saveDir, 'crop_img_hor')
    save_dir_ver = os.path.join(saveDir, 'crop_img_ver')
    if not os.path.exists(save_dir_hor):
        os.mkdir(save_dir_hor)
    if not os.path.exists(save_dir_ver):
        os.mkdir(save_dir_ver)
    fh = codecs.open(save_dir_hor + '/label.txt', 'w', encoding='utf-8')
    fv = codecs.open(save_dir_ver + '/label.txt', 'w', encoding='utf-8')

    horCount= verCount= 0
    for imgFile in imgFiles:
        img = Image.open(os.path.join(imgDir, imgFile)).convert('RGB')
        imgName, _= os.path.splitext(imgFile)
        txtName= imgName + '.txt'
        with open(os.path.join(txtDir, txtName), 'r', encoding="utf-8") as f:
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
                name = imgName + '_' + str(j) + '.jpg'
                if w > h:  # 横的图片
                    # 对image_new做resize
                    horCount += 1
                    p = img_new.size[1] / 31
                    # if p < 1:
                    #     continue
                    new_height = 31
                    new_width = int(img_new.size[0] / p)
                    log.debug('horizonal image: with width %+3s and height %+3s' % (new_width, new_height))
                    image_resized = img_new.resize((new_width, new_height))

                    image_resized.save(os.path.join(save_dir_hor, name))
                    fh.write(name + ' ' + label + '\n')
                else:  # 竖的图片
                    verCount += 1
                    p = img_new.size[0] / 31
                    # if p < 1:
                    #     continue
                    new_height = int(img_new.size[1] / p)
                    new_width = 31
                    log.debug('vertical image: with width %+3s and height %+3s' % (new_width, new_height))
                    image_resized = img_new.resize((new_width, new_height))

                    image_resized.save(os.path.join(save_dir_ver, name))
                    fv.write(name + ' ' + label + '\n')
        # break
    fh.close()
    fv.close()
    sumCount= horCount + verCount
    log.info('%s images cropped in total with %s horizational images and %s vertical images' %(sumCount, horCount, verCount))


def resize(imgDir, saveDir):
    root_dir = '../data/originData/image_val'
    save_dir_hor = os.path.join(saveDir, 'img_hor_resized')
    save_dir_ver = os.path.join(saveDir, 'img_ver_resized')
    if not os.path.exists(save_dir_hor):
        os.makedirs(save_dir_hor)
    if not os.path.exists(save_dir_ver):
        os.makedirs(save_dir_ver)

    for i, imgName in enumerate(os.listdir(imgDir)):
        log.info("%s'th image opened" %(i, ))
        img = Image.open(os.path.join(imgDir, imgName)).convert('RGB')
        if img.size[0] > img.size[1]: # 横的图片
            p = img.size[1] / 31
            # if p < 1:
            #     continue
            new_height = 31
            new_width = int(img.size[0] / p)
            log.debug('horizonal image: with width %+3s and height %+3s' % (new_width, new_height))
            image_resized = img.resize((new_width, new_height))
            image_resized.save(os.path.join(save_dir_hor, imgName))
        else:
            p = img.size[0] / 31
            # if p < 1:
            #     continue
            new_height = int(img.size[1] / p)
            new_width = 31
            log.debug('vertical image: with width %+3s and height %+3s' % (new_width, new_height))
            image_resized = img.resize((new_width, new_height))

            image_resized.save(os.path.join(save_dir_ver, imgName))
    log.info("Image Resize succeed! %s in total." %(i+ 1, ))


if __name__ == '__main__':
    preprocess()
    # resize("../data/originData/icpr_mtwi_task1/test_line_image")