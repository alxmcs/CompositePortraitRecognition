import cv2
import numpy as np
from arcface.lib import l2_norm, ArcFaceModel


def calculate_embedding_with_model(path, input_size, model):
    img = cv2.imread(path)
    img = convert_image(img, input_size)
    return l2_norm(model(img))


def calculate_embedding(path, input_size=300):
    img = cv2.imread(path)
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)
    img = convert_image(img, input_size)
    return l2_norm(model(img))


def calculate_embeddings(img, image_size, model):
    img = convert_image(img, image_size)
    embeds = l2_norm(model(img))
    return embeds


def get_distance(embeds1, embeds2):
    diff = np.subtract(embeds1, embeds2)
    distance = np.sum(np.square(diff))
    return distance


def convert_image(img, input_size):
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    return img


def calculate_distance(path1, path2, image_size, model):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    embeddings1 = calculate_embeddings(img1, image_size, model)
    embeddings2 = calculate_embeddings(img2, image_size, model)

    return get_distance(embeddings1, embeddings2)
