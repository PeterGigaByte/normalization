import os

import PIL
import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from wand.image import Image as wandImage

path = "C:\\Users\\barad\\PycharmProjects\\TP\\scrabble-gan\\res\\data\\iamDB\\words-Reading\\"
pathAugmentation = "C:\\Users\\barad\\PycharmProjects\\normalization-tp\\res\\"
bucket_size = 17

#path = "pictures\\"
#pathAugmentation = "pokus"
#bucket_size = 15


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


def applyEffect(ver, directory):
    save_path = pathAugmentation + "\\augmentation\\" + directory + "\\" + ver + "\\"
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
            save_file = save_path + "\\" + str(i) + "\\" + file
            image = np.asarray(image)
            # Histogram equ
            if ver == "Hver1":
                image = cv2.equalizeHist(image)
            if ver == "Hver2":
                image = exposure.equalize_adapthist(image, clip_limit=0.90)
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
            # Gaussian blur
            if ver == "Gver1":
                image = cv2.GaussianBlur(image, (1, 1), 1)
            if ver == "Gver2":
                image = cv2.GaussianBlur(image, (1, 1), 2)
            if ver == "Gver3":
                image = cv2.GaussianBlur(image, (1, 1), 3)
            if ver == "Gver4":
                image = cv2.GaussianBlur(image, (3, 3), 1)
            if ver == "Gver5":
                image = cv2.GaussianBlur(image, (3, 3), 2)
            if ver == "Gver6":
                image = cv2.GaussianBlur(image, (3, 3), 3)
            if ver == "Gver7":
                image = cv2.GaussianBlur(image, (3, 3), 4)
            # Sharpness
            if ver == "Sver1":
                with wandImage(filename=file_path) as img:
                    img.sharpen(radius = 8, sigma = 4)
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


def imageAugmentation():
    createDirectories()
    # histogram eq
    # applyEffect("Hver1", "histogram_eq")
    # applyEffect("Hver2", "histogram_eq")
    # applyEffect("Hver3", "histogram_eq")
    # applyEffect("Hver4", "histogram_eq")
    # applyEffect("Hver5", "histogram_eq")
    # applyEffect("Hver6", "histogram_eq")
    # applyEffect("Hver7", "histogram_eq")

    # gaussian blur
    # applyEffect("Gver1", "blur")
    # applyEffect("Gver2", "blur")
    # applyEffect("Gver3", "blur")
    # applyEffect("Gver4", "blur")
    # applyEffect("Gver5", "blur")
    # applyEffect("Gver6", "blur")
    # applyEffect("Gver7", "blur")

    # gaussian_noise
    #applyEffect("Nver1", "gaussian_noise")
    #applyEffect("Nver2", "gaussian_noise")
    #applyEffect("Nver3", "gaussian_noise")
    #applyEffect("Nver4", "gaussian_noise")
    #applyEffect("Nver5", "gaussian_noise")
    #applyEffect("Nver6", "gaussian_noise")
    #applyEffect("Nver7", "gaussian_noise")

    # sharpen
    applyEffect("Sver1", "sharpen")
    applyEffect("Sver2", "sharpen")
    applyEffect("Sver3", "sharpen")
    applyEffect("Sver4", "sharpen")
    applyEffect("Sver5", "sharpen")
    applyEffect("Sver6", "sharpen")
    applyEffect("Sver7", "sharpen")



if __name__ == "__main__":
    # deleteNoWords()
    # resizeImages()
    imageAugmentation()
