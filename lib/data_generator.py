"""

CE MODULE EST CHARGÉ DE FOURNIR LE DATASET

"""

import numpy as np
import cv2
import wget
from zipfile import ZipFile
import os

DATASET_URL = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/18648/1026645/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587295730&Signature=ksJUyTw5SwzscEu8o0n8IVEBFFPudl6sg79pFjclamPPobZTdlqz9%2FgP1ZTEmfOT6PcSIyJJ2bSpgc8k5fxreeMl1pa%2FNLIS1qhLGaGM%2BL9hUfMfzCIIX7vjHviIRmgBiGKKJ6VumchxykwhiJP4wAx%2BwDVmghT91%2FB%2BnJFwqaTfpDhonvpIjB0L2jhX8DX5J0wPGEiGjhlit9A8LD%2FUdMBAu17eEo20cGoZLVqCQ0hr3ZxRX9O%2BuuMH3AsFMkoye%2F%2B8XI63u4EJXJVsJPHk%2BcI6rafJ%2Bd6fKdzE%2Bd6EAyk71%2FAl6ovWOn6uIgoFAoTgw5Hzz%2FN64N%2FEG1XVjwBN%2BA%3D%3D&response-content-disposition=attachment%3B+filename%3Dplant-pathology-2020-fgvc7.zip"
DATASET_TARGET = "./data/"
TRAIN_CSV_PATH = "./data/train.csv"
IMG_PATH = "./data/images/"
IMG_EXT = ".jpg"
TRAIN_PROPORTION = 0.8


# récupère le dataset sur kaggle
def download_dataset():

    print("Plant Health dataset downloading.")

    try:
        os.mkdir(DATASET_TARGET)
        print("Creation of {}.".format(DATASET_TARGET))
    except FileExistsError as e:
        print("Directory {} already exist.".format(DATASET_TARGET))

    print("Downloading dataset from kaggle.")
    filename = wget.download(DATASET_URL, DATASET_TARGET)

    print("Extract dataset of {}.".format(filename))
    with ZipFile(filename) as zf:
        zf.extractall(DATASET_TARGET)

    os.remove(filename)


# retourne le nom des échantillojn avec leur pathologie
def get_all_dataset(train_path):
    try:

        print("Load plant image names and pathologies.")
        file = open(train_path, "r")
        lines = file.readlines()
        lines = lines[1:]
        img_label = []
        expectations = []

        i = 0
        while i < len(lines):
            lines[i] = lines[i].replace("\n", "")
            row = lines[i].split(",")
            pathology = [row[1], row[2], row[3], row[4]]
            img_label.append(row[0])
            expectations.append(pathology.index("1"))
            i += 1

        return img_label, expectations

    except Exception as e:

        print(e)


# retourne les images à partir de leurs noms
def get_img(img_label, img_path):
    try:

        print("Load plant images.")
        imgs = []

        for label in img_label:
            img_np = cv2.imread(img_path + label + IMG_EXT)
            imgs.append(img_np)

        return imgs

    except Exception as e:

        print(e)


# forme le dataset
def get_final_dataset(imgs, expectations):
    try:

        print("Creation of the final dataset.")
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        nb_train_data = int(len(imgs) * TRAIN_PROPORTION)

        for i in range(0, len(imgs)):

            if i < nb_train_data:
                x_train.append(imgs[i])
                y_train.append(expectations[i])

            else:
                x_test.append(imgs[i])
                y_test.append(expectations[i])

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    except Exception as e:

        print(e)


# fonction a appelé pour charger le dataset
def load_dataset():
    download_dataset()
    img_lab, expectations = get_all_dataset(TRAIN_CSV_PATH)
    imgs = get_img(img_lab, IMG_PATH)
    return get_final_dataset(imgs, expectations)


x_t, x_te, y_t, y_te = load_dataset()
print(x_t.shape, x_te.shape, y_t.shape, y_te.shape)