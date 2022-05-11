import os.path

import cv2
from arcface.lib import ArcFaceModel
from arcface import ArcFace
import my_arcface.main
import numpy as np


def get_best_distanse(embeds):
    # больше точно не будет
    best_distance = 100
    path_best_embeds = -1
    "C:\CompositePortraitRecongnition\output_embeds\1.npy"
    for i in range(1, 19):
        name = str(i) + ".npy"
        path = os.path.join("C:\CompositePortraitRecongnition", "output_embeds", name)
        embeds1 = np.load(path)
        diff = np.subtract(embeds1, embeds)
        distance = np.sum(np.square(diff))
        if distance < best_distance:
            best_distance = distance
            path_best_embeds = path

    return best_distance, path_best_embeds


if __name__ == "__main__":
    # CONSTANTS
    COUNT_FRAMES = 5
    INPUT_SIZE = 112

    # почему-то если путь к конфигу не абсолютный, то его не воспринимают и это все не работает
    cascPath = "C:\\CompositePortraitRecongnition\\configs\\cascade.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # for calculating embeddings
    input_size = INPUT_SIZE
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)



    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('C:\\CompositePortraitRecongnition\\video\\test.mp4')
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("C:\\CompositePortraitRecongnition\\dataset\\frame%d.jpg" % count, gray[y:y + h, x:x + w])
            count += 1
            embeds = my_arcface.main.calculate_embeddings(frame, model, input_size)
            best_distance, path_best_embeds = get_best_distanse(embeds)
            print(best_distance, path_best_embeds)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if count >= COUNT_FRAMES:
            break

        # to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
