import os
from datetime import datetime

import openpyxl
from arcface.lib import ArcFaceModel

import utils.my_arcface.main
import utils.tensorflow.style_transfer
import utils.tensorflow.face_encoding

if __name__ == "__main__":

    # tensorflow model
    transfer_model = utils.tensorflow.style_transfer.TransferModel(utils.tensorflow.style_transfer.MODEL_URL)

    headers = ['№ пары изображений',
               'Расстояние до переноса стиля tensorflow',
               'Расстояние после переноса стиля tensorflow',
               'Расстояние до переноса стиля arcface',
               'Расстояние после переноса стиля arcface']

    # before style_transfer
    arcface_encodings_array_before = []
    tensorflow_encodings_array_before = []
    # after style_transfer
    arcface_encodings_array_after = []
    tensorflow_encodings_array_after = []
    # array of indexes of successful operations
    successful_indexes = []

    input_size = 300
    # arcface model
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)

    for i in range(1, 20):
        print(f"{datetime.now()}: iteration number {i}")
        path0 = os.path.join("../images", "photos", f"photo{str(i + 1)}.png")
        path1 = os.path.join("../images", "sketches", f"sketch{str(i + 1)}.png")
        try:
            tensorflow_distance = utils.tensorflow.face_encoding.calculate_distance(path0, path1)
            arcface_distance = utils.my_arcface.main.calculate_distance(path0, path1, input_size, model)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии до переноса стиля")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue
        path_image_with_style = os.path.join("../images", "with_style", f"{str(i + 1)}.png")
        image_with_style = transfer_model.process_image(path0, path1, path_image_with_style)
        try:
            new_tensorflow_distance = utils.tensorflow.face_encoding.calculate_distance(path_image_with_style, path1)
            new_arcface_distance = utils.my_arcface.main.calculate_distance(path_image_with_style, path1, input_size,
                                                                            model)
        except IndexError as e:
            print(f"{str(e)} \n не удалось обнаружить лицо на фотографии после переноса стиля")
            continue
        tensorflow_encodings_array_before.append(tensorflow_distance)
        arcface_encodings_array_before.append(arcface_distance)
        tensorflow_encodings_array_after.append(new_tensorflow_distance)
        arcface_encodings_array_after.append(new_arcface_distance)
        print(f"{datetime.now()}: {tensorflow_distance}")
        print(f"{datetime.now()}: {new_tensorflow_distance}")
        successful_indexes.append(i)

    book = openpyxl.Workbook()
    sheet_1 = book.create_sheet("results", 0)
    sheet_1.append(headers)

    # в таблицу попадают только успешные преобразования стиля(в которых рассчиталось расстояние до и после)
    for i in range(0, len(successful_indexes)):
        sheet_1.append(
            [successful_indexes[i], tensorflow_encodings_array_before[i], tensorflow_encodings_array_after[i],
             arcface_encodings_array_before[i], arcface_encodings_array_after[i]])
    book.save("C:\\CompositePortraitRecongnition\\results\\results.xlsx")

    # this part calculates average gain for tensorflow
    avg_before_tensorflow = sum(tensorflow_encodings_array_before) / len(tensorflow_encodings_array_before)
    avg_after_tensorflow = sum(tensorflow_encodings_array_after) / len(tensorflow_encodings_array_after)
    print(
        f"{datetime.now()}: avg_before tensorflow: {avg_before_tensorflow}\navg_after tensorflow: {avg_after_tensorflow}")
    print(f"{datetime.now()}: improve tensorflow: {avg_before_tensorflow / avg_after_tensorflow}")

    # this part calculates average gain for arcface
    avg_before_arcface = sum(arcface_encodings_array_before) / len(arcface_encodings_array_before)
    avg_after_arcface = sum(arcface_encodings_array_after) / len(arcface_encodings_array_after)
    print(f"{datetime.now()}: avg_before arcface: {avg_before_arcface}\navg_after arcface: {avg_after_arcface}")
    print(f"{datetime.now()}: improve arcface: {avg_before_arcface / avg_after_arcface}")
