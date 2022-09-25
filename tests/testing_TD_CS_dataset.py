import os
from datetime import datetime

import mkl_random
import openpyxl
from PIL import Image
from arcface.lib import ArcFaceModel

import utils.my_arcface.main
import utils.tensorflow.face_encoding
import utils.tensorflow.style_transfer

if __name__ == "__main__":
    arcface_encodings = []
    wrong_arcface_encodings = []
    tensorflow_encodings = []
    wrong_tensorflow_encodings = []
    input_size = 300
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)

    for i in range(1, 113):
        print(f"{datetime.now()}: iteration number {i}")
        path1 = os.path.join("../dataset", "TDCS", str(i), f"TD_CS_{str(i)}.jpg")
        path0 = os.path.join("../dataset", "TDCS", str(i), "TD_RGB_E_1.jpg")
        random_sketch_index = mkl_random.randint(1, 113)
        path2 = os.path.join("../dataset", "TDCS", str(random_sketch_index), "TD_RGB_E_1.jpg")
        try:
            portrait_image = Image.open(path0)
            portrait_image.thumbnail((input_size, input_size))
            portrait_image.save("portrait_resized.png")

            sketch_image = Image.open(path1)
            sketch_image.thumbnail((input_size, input_size))
            sketch_image.save('sketch_resized.png')

            random_sketch_image = Image.open(path2)
            random_sketch_image.thumbnail((input_size, input_size))
            random_sketch_image.save('random_sketch_resized.png')

            tensorflow_distance = utils.tensorflow.face_encoding.calculate_distance("portrait_resized.png",
                                                                                    'sketch_resized.png')
            wrong_tensorflow_distance = utils.tensorflow.face_encoding.calculate_distance("portrait_resized.png",
                                                                                          'random_sketch_resized.png')

            arcface_distance = utils.my_arcface.main.calculate_distance("portrait_resized.png", 'sketch_resized.png',
                                                                        input_size, model)
            wrong_arcface_distance = utils.my_arcface.main.calculate_distance("portrait_resized.png",
                                                                              'random_sketch_resized.png',
                                                                              input_size, model)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue
        arcface_encodings.append(arcface_distance)
        tensorflow_encodings.append(tensorflow_distance)
        wrong_arcface_encodings.append(wrong_arcface_distance)
        wrong_tensorflow_encodings.append(wrong_tensorflow_distance)

    avg_arcface = sum(arcface_encodings) / len(arcface_encodings)
    avg_tensorflow = sum(tensorflow_encodings) / len(tensorflow_encodings)
    avg_wrong_arcface = sum(wrong_arcface_encodings) / len(wrong_arcface_encodings)
    avg_wrong_tensorflow = sum(wrong_tensorflow_encodings) / len(wrong_tensorflow_encodings)

    print(f"avg_arcface is {avg_arcface} ")
    print(f"avg_tensorflow is {avg_tensorflow} ")
    print(f"avg_wrong_arcface is {avg_wrong_arcface} ")
    print(f"avg_wrong_tensorflow is {avg_wrong_tensorflow} ")

    book = openpyxl.Workbook()
    headers = ['tensorflow distance',
               'arcface distance',
               'wrong tensorflow distance',
               'wrong arcface distance']
    sheet_1 = book.create_sheet("results", 0)
    sheet_1.append(headers)
    sheet_1.append([avg_tensorflow, avg_arcface, avg_wrong_tensorflow, avg_wrong_arcface])
    book.save("C:\\CompositePortraitRecongnition\\tests\\testing_TDCS_results.xlsx")
