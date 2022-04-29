import cv2
import numpy as np
from absl.flags import FLAGS
from arcface import ArcFace
from arcface.lib import ArcFaceModel, l2_norm

if __name__ == "__main__":

    input_size = 112
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)

    img = cv2.imread('C:\\CompositePortraitRecongnition\\images\\photos\\photo1.png')
    if img is None:
        print("failed imread")
    else:
        img = cv2.resize(img, (input_size, input_size))
        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        embeds = l2_norm(model(img))
        print(embeds)


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



