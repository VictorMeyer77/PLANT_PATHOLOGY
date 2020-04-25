"""

CE MODULE GÉNÈRE LE FICHIER DE TEST DE MODÈLE
LE FORMAT EST BON POUR LE PUBLIER SUR KAGGLE

"""

import cv2
import numpy as np

DATASET_TARGET = "./data/"
TEST_CSV_PATH = "./data/test.csv"
IMG_PATH = "./data/images/"
IMG_EXT = ".jpg"
COLUMNS_TEST_LABEL = "image_id,healthy,multiple_diseases,rust,scab\n"
OUTPUT = "./output/"


# récupère le nom des images de test
def get_img_test_names(test_path):
    try:

        print("Load test images names.")
        file = open(test_path, "r")
        lines = file.readlines()
        name_lines = lines[1:]
        names = []

        for line in name_lines:
            names.append(line.replace("\n", ""))

        return names

    except Exception as e:

        print(e)


# retourne les images à partir de leurs noms
def get_test_imgs(test_labels, img_path, img_size):
    try:

        print("Load plant test images.")
        imgs = []

        for label in test_labels:
            img_np = cv2.imread(img_path + label + IMG_EXT)
            img_np = cv2.resize(img_np, img_size)
            imgs.append(img_np)

        return np.array(imgs) / 255

    except Exception as e:

        print(e)


# retourne une liste de prdictions
def generate_predictions(test_imgs, model):
    try:
        print("Genrerate predictions.")
        predicts = np.argmax(model.predict(test_imgs), axis=1).tolist()
        return predicts

    except Exception as e:

        print(e)


# crée le fichier à livrer sur Kaggle
def generate_predict_file(output_dir, output_name, predictions, col_labels, img_test_labels):
    try:

        file = open(output_dir + output_name, "w")
        file.write(col_labels)

        for i in range(0, len(predictions)):
            row = [0] * 4
            row[predictions[i]] = 1
            file.write(img_test_labels[i] + "," + str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(
                row[3]) + "\n")

        file.close()
        print("Generate answer file: {}.".format(output_dir + output_name))
    except Exception as e:

        print(e)


def prediction(img_size, model, output_name):
    test_img_label = get_img_test_names(TEST_CSV_PATH)
    test_imgs = get_test_imgs(test_img_label, IMG_PATH, img_size)
    pred = generate_predictions(test_imgs, model)
    generate_predict_file(OUTPUT, output_name, pred, COLUMNS_TEST_LABEL, test_img_label)
