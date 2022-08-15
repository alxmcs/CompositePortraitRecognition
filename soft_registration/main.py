import json
import cv2
import numpy as np
import utils.tensorflow.face_encoding
from skimage.io import imread, imsave, imshow
from PIL import Image
import openpyxl
from arcface.lib import ArcFaceModel

import utils.my_arcface.main
import utils.tensorflow.style_transfer
import utils.tensorflow.face_encoding

if __name__ == "__main__":

    # settings = {'portrait_path': 'C:\\CompositePortraitRecongnition\\images\\photos\\photo1.png',
    #             'sketch_path': 'C:\\CompositePortraitRecongnition\\images\\sketches\\sketch1.png',
    #             'threshold': 0.5
    #             }
    #
    # with open('settings.json', 'w') as fp:
    #     json.dump(settings, fp)

    with open('settings.json') as info_data:
        json_data = json.load(info_data)

    portrait_path = json_data['portrait_path']
    sketch_path = json_data['sketch_path']
    threshold = json_data['threshold']

    distance = utils.tensorflow.face_encoding.calculate_distance(portrait_path, sketch_path)
    result = False
    if distance < threshold:
        result = True
    settings = {'tensorflow_distance': distance, 'is_like': result}
    print(settings['tensorflow_distance'])
    print(settings['is_like'])

    input_size = 300
    portrait_image = Image.open(portrait_path)
    print(f"Original size : {portrait_image.size}")  # 5464x3640

    portrait_image.thumbnail((input_size, input_size))
    portrait_image.save("portrait_resized.png")

    sketch_image = Image.open(sketch_path)
    print(f"Original size : {sketch_image.size}")  # 5464x3640

    sketch_image.thumbnail((input_size, input_size))
    sketch_image.save('sketch_resized.png')

    # arcface model
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)
    distance = utils.my_arcface.main.calculate_distance('portrait_resized.png', 'sketch_resized.png', input_size, model)
    result = False
    if distance < threshold:
        result = True
    settings = {'arcface_distance': distance, 'is_like': result}
    print(settings['arcface_distance'])
    print(settings['is_like'])
