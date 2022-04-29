import os
from datetime import datetime

import cv2
import numpy as np
from absl.flags import FLAGS
from arcface import ArcFace
from arcface.lib import ArcFaceModel, l2_norm


def convert_image(img):
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    return img


if __name__ == "__main__":

    input_size = 112
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)

    for i in range(1, 20):
        print(f"{datetime.now()}: iteration number {i}")
        # C:\CompositePortraitRecongnition\images\photos\photo1.png
        path0 = os.path.join("C:\\CompositePortraitRecongnition", "images", "photos", f"photo{str(i + 1)}.png")
        path1 = os.path.join("C:\\CompositePortraitRecongnition", "images", "sketches", f"sketch{str(i + 1)}.png")
        img0 = cv2.imread(path0)
        img1 = cv2.imread(path1)
        if img0 is None:
            print("error reading image 0")
        else:
            img0 = convert_image(img0)
            embeds0 = l2_norm(model(img0))
        if img1 is None:
            print("error reading image 1")
        else:
            img1 = convert_image(img1)
            embeds1 = l2_norm(model(img1))
            out_path = os.path.join("C:\\CompositePortraitRecongnition", "output_embeds", str(i))
            np.save(out_path, embeds1)

# Эта часть почему-то не работает ...
# face_rec = ArcFace.ArcFace()
# emb1 = face_rec.calc_emb("images//photos//photo1.png")
# print(emb1)
# emb2 = face_rec.calc_emb("images//sketches//sketch1.png")
# print(emb2)
# distance = face_rec.get_distance_embeddings(emb1, emb2)
# print(f"distance = ${distance}")

# Я пока не понял, в чем тут дело...ему интерпретатор какой-то не нравится?
# Traceback (most recent call last):
#   File "C:\CompositePortraitRecongnition\arcface\main.py", line 28, in <module>
#     face_rec = ArcFace.ArcFace()
#   File "C:\Anaconda\envs\new_environment\lib\site-packages\arcface\ArcFace.py", line 39, in __init__
#     self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
#   File "C:\Anaconda\envs\new_environment\lib\site-packages\tensorflow\lite\python\interpreter.py", line 365, in __init__
#     _interpreter_wrapper.CreateWrapperFromFile(
# ValueError: Could not open 'C:\Users\Дом\.astropy\cache\download\url\df64a54c38e4116a7f961668f6e12439\contents'.
