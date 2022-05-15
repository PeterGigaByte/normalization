import os
import shutil

import PIL
import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from wand.image import Image as wandImage
from numba import cuda, jit

path2 = "C:\\Users\\barad\\PycharmProjects\\normalization-tp\\pictures\\"
path = "C:\\Users\\barad\\PycharmProjects\\TP\\scrabble-gan\\res\\data\\iamDB\\words-Reading\\"
pathAugmentation = "C:\\Users\\barad\\PycharmProjects\\normalization-tp\\res\\"
bucket_size = 17


# path = "pictures\\"
# pathAugmentation = "pokus"
# bucket_size = 15


def deleteNoWords():
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(1, bucket_size + 1, 1):
        try:
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
        except:
            print("Folder " + str(i) + "not exist.")


def resizeImages():
    w = 16
    h = 32
    for i in range(1, bucket_size + 1, 1):
        try:
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
        except:
            print("Folder " + str(i) + "not exist.")


def createDirectories():
    dirs = ["\\histogram_eq", "\\gaussian_noise", "\\sharpen", "\\blur"]
    for dir in dirs:
        if not os.path.exists(pathAugmentation + "\\augmentation" + dir):
            os.makedirs(pathAugmentation + "\\augmentation" + dir)


def createTextDocument(textPath):
    text = os.path.splitext(textPath)
    text = text[0] + ".txt"
    with open(text) as f:
        lines = f.read()  ##Assume the sample file has 3 lines
        first = lines.split('\n', 1)[0]
    return first, os.path.splitext(text)[1], text


def applyEffect(ver, directory, copy=False):
    save_path = pathAugmentation + "\\augmentation\\" + directory + "\\" + ver + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(1, bucket_size + 1, 1):
        try:
            reading_dir = path + str(i) + '\\'
            file_list = os.listdir(reading_dir)
            file_list = [fi for fi in file_list if fi.endswith(".png")]
            for file in file_list:
                file_path = reading_dir + file
                word = createTextDocument(file_path)
                image = Image.open(file_path)
                createTextDocument(file_path)
                if not os.path.exists(save_path + "\\" + str(i)):
                    os.makedirs(save_path + "\\" + str(i))
                save_file = save_path + "\\" + str(i) + "\\" + ver + file
                image = np.asarray(image)
                if copy:
                    if "H" not in file and "S" not in file and "N" not in file and "G" not in file:
                        save_file = save_path + str(i) + "\\" + ver + file
                        save_text = os.path.splitext(save_file)[0] + word[1]
                        head_tail = os.path.split(file_path)
                        photoDST = head_tail[0] + "\\" + ver + head_tail[1]
                        head_tail = os.path.split(word[2])
                        txtDST = head_tail[0] + "\\" + ver + head_tail[1]
                        photoSRC = save_file
                        txtSRC = save_text
                        if os.path.exists(photoDST):
                            os.remove(photoDST)
                        if os.path.exists(photoDST):
                            os.remove(txtDST)
                        shutil.copy(photoSRC, photoDST)
                        shutil.copy(txtSRC, txtDST)
                else:
                    # Histogram equ
                    if ver == "Hver1":
                        p2, p98 = np.percentile(image, (8, 92))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver2":
                        p2, p98 = np.percentile(image, (10, 90))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver3":
                        p2, p98 = np.percentile(image, (2, 98))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver4":
                        p2, p98 = np.percentile(image, (15, 85))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver5":
                        p2, p98 = np.percentile(image, (12, 88))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver6":
                        p2, p98 = np.percentile(image, (5, 95))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    if ver == "Hver7":
                        p2, p98 = np.percentile(image, (7, 83))
                        image = exposure.rescale_intensity(image, in_range=(p2, p98))
                    # Gaussian blur
                    if ver == "Gver1":
                        image = cv2.GaussianBlur(image, (1, 1), 0.5)
                    if ver == "Gver2":
                        image = cv2.GaussianBlur(image, (2, 2), 0.7)
                    if ver == "Gver3":
                        image = cv2.GaussianBlur(image, (2, 2), 0.8)
                    if ver == "Gver4":
                        image = cv2.GaussianBlur(image, (2, 2), 0.9)
                    if ver == "Gver5":
                        image = cv2.GaussianBlur(image, (3, 3), 1)
                    if ver == "Gver6":
                        image = cv2.GaussianBlur(image, (3, 3), 1.1)
                    if ver == "Gver7":
                        image = cv2.GaussianBlur(image, (1, 1), 1.2)
                    # Sharpness
                    if ver == "Sver1":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=8, sigma=4)
                            img.save(filename=save_file)
                    if ver == "Sver2":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=16, sigma=4)
                            img.save(filename=save_file)
                    if ver == "Sver3":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=16, sigma=8)
                            img.save(filename=save_file)
                    if ver == "Sver4":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=16, sigma=16)
                            img.save(filename=save_file)
                    if ver == "Sver5":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=32, sigma=8)
                            img.save(filename=save_file)
                    if ver == "Sver6":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=32, sigma=16)
                            img.save(filename=save_file)
                    if ver == "Sver7":
                        with wandImage(filename=file_path) as img:
                            img.sharpen(radius=32, sigma=32)
                            img.save(filename=save_file)
                    # Noise
                    if ver == "Nver1":
                        with wandImage(filename=file_path) as img:
                            img.noise("laplacian", attenuate=0.8)
                            img.save(filename=save_file)
                    if ver == "Nver2":
                        with wandImage(filename=file_path) as img:
                            img.noise("laplacian", attenuate=1.0)
                            img.save(filename=save_file)
                    if ver == "Nver3":
                        with wandImage(filename=file_path) as img:
                            img.noise("gaussian", attenuate=1.0)
                            img.save(filename=save_file)
                    if ver == "Nver4":
                        with wandImage(filename=file_path) as img:
                            img.noise("gaussian", attenuate=0.5)
                            img.save(filename=save_file)
                    if ver == "Nver5":
                        with wandImage(filename=file_path) as img:
                            img.noise("gaussian", attenuate=0.8)
                            img.save(filename=save_file)
                    if ver == "Nver6":
                        with wandImage(filename=file_path) as img:
                            img.noise("gaussian", attenuate=0.7)
                            img.save(filename=save_file)
                    if ver == "Nver7":
                        with wandImage(filename=file_path) as img:
                            img.noise("gaussian", attenuate=0.6)
                            img.save(filename=save_file)
                    if "N" or "S" not in ver:
                        cv2.imwrite(save_file, image)
                    pathText = os.path.splitext(save_file)[0] + word[1]
                    f = open(pathText, "w+")
                    f.write(word[0])
                    f.close()
        except:
            print("Folder " + str(i) + "not exist.")


def histogram_eq(copy):
    applyEffect("Hver1", "histogram_eq", copy)
    applyEffect("Hver2", "histogram_eq", copy)
    applyEffect("Hver3", "histogram_eq", copy)
    applyEffect("Hver4", "histogram_eq", copy)
    applyEffect("Hver5", "histogram_eq", copy)
    applyEffect("Hver6", "histogram_eq", copy)
    applyEffect("Hver7", "histogram_eq", copy)


def gaussian_blur(copy):
    applyEffect("Gver1", "blur", copy)
    applyEffect("Gver2", "blur", copy)
    applyEffect("Gver3", "blur", copy)
    applyEffect("Gver4", "blur", copy)
    applyEffect("Gver5", "blur", copy)
    applyEffect("Gver6", "blur", copy)
    applyEffect("Gver7", "blur", copy)


def gaussian_noise(copy):
    applyEffect("Nver1", "gaussian_noise", copy)
    applyEffect("Nver2", "gaussian_noise", copy)
    applyEffect("Nver3", "gaussian_noise", copy)
    applyEffect("Nver4", "gaussian_noise", copy)
    applyEffect("Nver5", "gaussian_noise", copy)
    applyEffect("Nver6", "gaussian_noise", copy)
    applyEffect("Nver7", "gaussian_noise", copy)


def sharpen(copy):
    applyEffect("Sver1", "sharpen", copy)
    applyEffect("Sver2", "sharpen", copy)
    applyEffect("Sver3", "sharpen", copy)
    applyEffect("Sver4", "sharpen", copy)
    applyEffect("Sver5", "sharpen", copy)
    applyEffect("Sver6", "sharpen", copy)
    applyEffect("Sver7", "sharpen", copy)


def imageAugmentation():
    createDirectories()
    copy = True
    histogram_eq(copy)
    gaussian_noise(copy)
    gaussian_blur(copy)
    sharpen(copy)


if __name__ == "__main__":
    deleteNoWords()
    resizeImages()
    imageAugmentation()
