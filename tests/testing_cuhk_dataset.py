import os
from datetime import datetime

import mkl_random
import openpyxl
from PIL import Image
from arcface.lib import ArcFaceModel

import utils.my_arcface.main
import utils.tensorflow.face_encoding
import utils.tensorflow.style_transfer

from tests.distance_visualization import display_results

if __name__ == "__main__":
    content1 = os.listdir('C:\\CompositePortraitRecongnition\\dataset\\CUHK\\testing_photos')
    # print(content1)
    # content2 = os.listdir('C:\\CompositePortraitRecongnition\\dataset\\CUHK\\testing_sketches')
    # print(content2)
    transfer_model = utils.tensorflow.style_transfer.TransferModel(
        utils.tensorflow.style_transfer.MODEL_URL)

    arcface_encodings_before = []
    wrong_arcface_encodings_before = []
    arcface_encodings_after = []
    wrong_arcface_encodings_after = []
    tensorflow_encodings_before = []
    tensorflow_encodings_after = []
    wrong_tensorflow_encodings_before = []
    wrong_tensorflow_encodings_after = []
    input_size = 300
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)
    i = 0
    for file in content1:
        print(f"{datetime.now()}: iteration number {i}")
        i += 1
        path1 = os.path.join("../dataset", "CUHK", "testing_photos", file)
        new_file = file.replace('.jpg', '-sz1.jpg')
        path0 = os.path.join("../dataset", "CUHK", "testing_sketches", new_file)
        random_sketch_index = mkl_random.randint(1, 113)
        path2 = os.path.join("../dataset", "TDCS", str(random_sketch_index), f"TD_CS_{str(random_sketch_index)}.jpg")
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

            path_image_with_style = os.path.join("../dataset", "CUHK", "with_style", f"CUHK_with_style_{str(i)}.jpg")
            path_image_with_random_style = os.path.join("../dataset", "CUHK", "with_style",
                                                        f"CUHK_with_random_style_{str(i)}.jpg")
            image_with_style = transfer_model.process_image("portrait_resized.png", "sketch_resized.png",
                                                            path_image_with_style)
            image_with_random_style = transfer_model.process_image("portrait_resized.png", "random_sketch_resized.png",
                                                                   path_image_with_random_style)

            tensorflow_distance_before = utils.tensorflow.face_encoding.calculate_distance("portrait_resized.png",
                                                                                           'sketch_resized.png')
            wrong_tensorflow_distance_before = utils.tensorflow.face_encoding.calculate_distance("portrait_resized.png",
                                                                                                 'random_sketch_resized.png')

            tensorflow_distance_after = utils.tensorflow.face_encoding.calculate_distance(path_image_with_style,
                                                                                          'sketch_resized.png')

            wrong_tensorflow_distance_after = utils.tensorflow.face_encoding.calculate_distance(
                path_image_with_random_style, 'random_sketch_resized.png')

            arcface_distance_before = utils.my_arcface.main.calculate_distance("portrait_resized.png",
                                                                               'sketch_resized.png',
                                                                               input_size, model)
            wrong_arcface_distance_before = utils.my_arcface.main.calculate_distance("portrait_resized.png",
                                                                                     'random_sketch_resized.png',
                                                                                     input_size, model)

            arcface_distance_after = utils.my_arcface.main.calculate_distance(path_image_with_style,
                                                                              'sketch_resized.png',
                                                                              input_size, model)
            wrong_arcface_distance_after = utils.my_arcface.main.calculate_distance(path_image_with_random_style,
                                                                                    'random_sketch_resized.png',
                                                                                    input_size, model)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue
        tensorflow_encodings_before.append(tensorflow_distance_before)
        wrong_tensorflow_encodings_before.append(wrong_tensorflow_distance_before)
        tensorflow_encodings_after.append(tensorflow_distance_after)
        wrong_tensorflow_encodings_after.append(wrong_tensorflow_distance_after)
        arcface_encodings_before.append(arcface_distance_before)
        wrong_arcface_encodings_before.append(wrong_arcface_distance_before)
        arcface_encodings_after.append(arcface_distance_after)
        wrong_arcface_encodings_after.append(wrong_arcface_distance_after)

    avg_tensorflow_before = sum(tensorflow_encodings_before) / len(tensorflow_encodings_before)
    avg_wrong_tensorflow_before = sum(wrong_tensorflow_encodings_before) / len(wrong_tensorflow_encodings_before)
    avg_tensorflow_after = sum(tensorflow_encodings_after) / len(tensorflow_encodings_after)
    avg_wrong_tensorflow_after = sum(wrong_tensorflow_encodings_after) / len(wrong_tensorflow_encodings_after)

    avg_arcface_before = sum(arcface_encodings_before) / len(arcface_encodings_before)
    avg_wrong_arcface_before = sum(wrong_arcface_encodings_before) / len(wrong_arcface_encodings_before)
    avg_arcface_after = sum(arcface_encodings_after) / len(arcface_encodings_after)
    avg_wrong_arcface_after = sum(wrong_arcface_encodings_after) / len(wrong_arcface_encodings_after)

    print(f"avg_tensorflow_before is {avg_tensorflow_before} ")
    print(f"avg_wrong_tensorflow_before is {avg_wrong_tensorflow_before} ")
    print(f"avg_tensorflow_after is {avg_tensorflow_after} ")
    print(f"avg_wrong_tensorflow_after is {avg_wrong_tensorflow_after} ")

    print(f"avg_arcface_before is {avg_arcface_before} ")
    print(f"avg_wrong_arcface_before is {avg_wrong_arcface_before} ")
    print(f"avg_arcface_after is {avg_arcface_after} ")
    print(f"avg_wrong_arcface_after is {avg_wrong_arcface_after} ")

    book = openpyxl.Workbook()
    headers = ['tensorflow distance before',
               'wrong tensorflow distance before',
               'tensorflow distance after ',
               'wrong tensorflow distance after',
               'arcface distance before',
               'wrong arcface distance before',
               'arcface distance after',
               'wrong arcface distance after']
    sheet_1 = book.create_sheet("results", 0)
    sheet_1.append(headers)
    sheet_1.append(
        [avg_tensorflow_before, avg_wrong_tensorflow_before, avg_tensorflow_after, avg_wrong_tensorflow_after,
         avg_arcface_before, avg_wrong_arcface_before, avg_arcface_after, avg_wrong_arcface_after])
    book.save("C:\\CompositePortraitRecongnition\\tests\\testing_CUHK_results.xlsx")
