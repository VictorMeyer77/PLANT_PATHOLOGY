"""

CE MODULE EST CHARGÉ DE FOURNIR LE DATASET
BIEN QUE LE REDIMENSIONNAGE DEVRAIT ÊTRE FAIT DANS LES MODÈLES
LA FAIBLE MÉMOIRE (8Go) DE MA MACHINE M'OBLIGE À RÉDIMENSIONNER EN CHARGEANT LE DATASET

"""

import numpy as np
import cv2
import wget
from zipfile import ZipFile
import os

DATASET_URL = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/18648/1026645/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587842427&Signature=rptfRIMs6hLQgGPxSXJtMEt1Su%2B%2B2R18idgODb5mFex%2Fztrz7FQA7MZLFcd1T37P1JA7fM3o8ubV3B6OKHx6a%2FKuGiBPMLZvDiQ78cRFdycZsbacKXRX2CMEujlAe1gr6n2Ico55sRnXr6oIK115Ed%2BxywEIrf9bCyaAxdaqPge8syBzqunsTdGSsX8iPpKFlh6Gq33vl7wuTNLcU4%2FVzBSaJLYpsTWMnMw3o2385Ex5E8eTELHw3oYpwQGFnJ6D3Q%2B9jjh8iWB62N0TBr8crwWp83QOK9Ugu8TFpyQ0Km%2FZCr06zkyCvh6dDGdqI%2FhyKpT42GUvNVejT9ewwHlkZA%3D%3D&response-content-disposition=attachment%3B+filename%3Dplant-pathology-2020-fgvc7.zip"
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
        if os.path.exists(DATASET_TARGET) and os.path.exists(TRAIN_CSV_PATH) and os.path.exists(IMG_PATH):
            print("Dataset already install.")
            return

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
def get_img(img_label, img_path, img_size):
    try:

        print("Load plant images.")
        imgs = []

        for label in img_label:
            img_np = cv2.imread(img_path + label + IMG_EXT)
            img_np = cv2.resize(img_np, img_size)
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
def load_dataset(img_size):
    download_dataset()
    img_lab, expectations = get_all_dataset(TRAIN_CSV_PATH)
    imgs = get_img(img_lab, IMG_PATH, img_size)
    return get_final_dataset(imgs, expectations)
