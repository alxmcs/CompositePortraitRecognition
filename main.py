import openpyxl

import utils.face_encoding
import utils.style_transfer
from openpyxl import load_workbook

if __name__ == "__main__":

    transfer_model = utils.style_transfer.TransferModel(utils.style_transfer.MODEL_URL)

    images = ["images\photos\photo1.png",
              "images\photos\photo2.png",
              "images\photos\photo3.png",
              "images\photos\photo4.png"]

    sketches = ["images\sketches\sketch1.png",
                "images\sketches\sketch2.png",
                "images\sketches\sketch3.png",
                "images\sketches\sketch4.png"]

    # before style_transfer
    encodings_array_before = []
    # after style_transfer
    encodings_array_after = []

    for i in range(0, len(images)):
        path0 = images[i]
        path1 = sketches[i]
        distance = utils.face_encoding.calculate_distance(path0, path1)
        encodings_array_before.append(distance)
        path_image_with_style = "images\with_style\\" + (i+1).__str__() + ".png"
        image_with_style = transfer_model.process_image(path0, path1, path_image_with_style)
        new_distance = utils.face_encoding.calculate_distance(path_image_with_style, path1)
        encodings_array_after.append(new_distance)
        print("iteration number " + i.__str__())

    # эта часть кода считает средний выигрыш
    # avg_before = sum(encodings_array_before)/len(encodings_array_before)
    # avg_after = sum(encodings_array_after)/len(encodings_array_after)
    #
    # print("avg_before " + avg_before.__str__() + "\n" + "avg_after " + avg_after.__str__())
    #
    # # насколько стало лучше
    # improve = avg_before/avg_after
    # print("improve = " + improve.__str__())

    headers = ['№ пары изображений', 'Расстояние до переноса стиля', 'Расстояние после переноса стиля']

    book = openpyxl.Workbook()
    sheet_1 = book.create_sheet("results", 0)
    sheet_1.append(headers)

    for i in range(0, len(encodings_array_before)):
        sheet_1.append([i + 1, encodings_array_before[i], encodings_array_after[i]])

    book.save("results/results.xlsx")

