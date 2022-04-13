import os

import PIL
import cv2
import numpy as np
from PIL import Image
from skimage import exposure

path = "C:\\Users\\barad\\PycharmProjects\\TP\\scrabble-gan\\res\\data\\iamDB\\words-Reading\\"
pathAugmentation = "C:\\Users\\barad\\PycharmProjects\\TP\\"
bucket_size = 17


def deleteNoWords():
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(1, bucket_size + 1, 1):
        reading_dir = path + str(i) + '\\'
        file_list = os.listdir(reading_dir)
        file_list = [fi for fi in file_list if fi.endswith(".txt")]
        for file in file_list:
            with open(reading_dir + file, 'r', encoding='utf8') as f:
                for char in f.readline():
                    if char not in char_vec:
                        f.close()
                        os.remove(reading_dir + file)
                        os.remove(os.path.splitext(reading_dir + file)[0] + '.png')
                        break


def resizeImages():
    w = 16
    h = 32
    for i in range(1, bucket_size + 1, 1):
        reading_dir = path + str(i) + '\\'
        file_list = os.listdir(reading_dir)
        file_list = [fi for fi in file_list if fi.endswith(".png")]
        for file in file_list:
            file_path = reading_dir + file
            image = Image.open(file_path)
            image = image.resize((int(w * i), int(h)), PIL.Image.ANTIALIAS)
            print(w * i, h, i, file)
            print("SAVE", file_path)
            image.save(file_path)


def createDirectories():
    dirs = ["\\histogram_eq", "\\gaussian_noise", "\\sharpen", "\\blur"]
    for dir in dirs:
        if not os.path.exists(pathAugmentation + "\\augmentation" + dir):
            os.makedirs(pathAugmentation + "\\augmentation" + dir)


def applyHistogramVer(ver):
    save_path = pathAugmentation + "\\augmentation\\histogram_eq\\" + ver + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(1, bucket_size + 1, 1):
        reading_dir = path + str(i) + '\\'
        file_list = os.listdir(reading_dir)
        file_list = [fi for fi in file_list if fi.endswith(".png")]
        for file in file_list:
            file_path = reading_dir + file
            image = Image.open(file_path)
            if not os.path.exists(save_path+"\\" + str(i)):
                os.makedirs(save_path+"\\" + str(i))
            file_path = save_path + "\\" + str(i) + "\\" + file
            image = np.asarray(image)
            if ver == "Hver1":
                image = cv2.equalizeHist(image)
            if ver == "Hver2":
                image = exposure.equalize_adapthist(image, clip_limit=0.03)
            if ver == "Hver3":
                p2, p98 = np.percentile(image, (2, 98))
                image = exposure.rescale_intensity(image, in_range=(p2, p98))
            if ver == "Hver4":
                return
            if ver == "Hver5":
                return
            if ver == "Hver6":
                return
            if ver == "Hver7":
                return
            cv2.imwrite(file_path, image)


def gaussian(ver):
    save_path = pathAugmentation + "\\augmentation\\gaussian\\" + ver + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(1, bucket_size + 1, 1):
        reading_dir = path + str(i) + '\\'
        file_list = os.listdir(reading_dir)
        file_list = [fi for fi in file_list if fi.endswith(".png")]
        for file in file_list:
            file_path = reading_dir + file
            image = Image.open(file_path)
            if not os.path.exists(save_path + "\\" + str(i)):
                os.makedirs(save_path + "\\" + str(i))
            file_path = save_path + "\\" + str(i) + "\\" + file
            image = np.asarray(image)
            if ver == "Gver1":
                image = cv2.GaussianBlur(image,(3,3),0)
            if ver == "Gver2":
                image = cv2.GaussianBlur(image,(3,3),1)
            if ver == "Gver3":
                image = cv2.GaussianBlur(image,(3,3),3)
            if ver == "Gver4":
                image = cv2.GaussianBlur(image,(5,5),0)
            if ver == "Gver5":
                image = cv2.GaussianBlur(image,(5,5),1)
            if ver == "Gver6":
                image = cv2.GaussianBlur(image,(5,5),3)
            if ver == "Gver7":
                image = cv2.GaussianBlur(image,(7,7),0)
            cv2.imwrite(file_path, image)


def imageAugmentation():
    createDirectories()
    applyHistogramVer("Hver1")
    applyHistogramVer("Hver2")
    applyHistogramVer("Hver3")
    applyHistogramVer("Hver4")
    applyHistogramVer("Hver5")
    applyHistogramVer("Hver6")
    applyHistogramVer("Hver7")
    gaussian("Gver1")
    gaussian("Gver2")
    gaussian("Gver3")
    gaussian("Gver4")
    gaussian("Gver5")
    gaussian("Gver6")
    gaussian("Gver7")


if __name__ == "__main__":
    # deleteNoWords()
    # resizeImages()
    imageAugmentation()
