import cv2
import numpy as np
from arcface.lib import l2_norm


def calculate_embeddings(img, model):
    embeds = l2_norm(model(img))
    return embeds


def get_distance(embeds1, embeds2):
    diff = np.subtract(embeds1, embeds2)
    distance = np.sum(np.square(diff))
    return distance


def convert_image(img, input_size):
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    return img


def calculate_distance(path1, path2, input_size, model):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1 = convert_image(img1, input_size)
    img2 = convert_image(img2, input_size)

    embeddings1 = calculate_embeddings(img1, model)
    embeddings2 = calculate_embeddings(img2, model)

    return get_distance(embeddings1, embeddings2)
