import os
import openpyxl
import utils.face_encoding
import utils.style_transfer
from datetime import datetime

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

    headers = ['№ пары изображений',
               'Расстояние до переноса стиля',
               'Расстояние после переноса стиля']

    # before style_transfer
    encodings_array_before = []
    # after style_transfer
    encodings_array_after = []

    for i in range(0, len(images)):
        path0 = images[i]
        path1 = sketches[i]
        distance = utils.face_encoding.calculate_distance(path0, path1)
        encodings_array_before.append(distance)
        path_image_with_style = os.path.join("images", "with_style", str(i + 1), ".png")
        image_with_style = transfer_model.process_image(path0, path1, path_image_with_style)
        new_distance = utils.face_encoding.calculate_distance(path_image_with_style, path1)
        encodings_array_after.append(new_distance)
        print(f"{datetime.now()}: iteration number {i}")
        print(f"{datetime.now()}: {distance}")
        print(f"{datetime.now()}: {new_distance}")

    # this part calculates average gain
    avg_before = sum(encodings_array_before) / len(encodings_array_before)
    avg_after = sum(encodings_array_after) / len(encodings_array_after)
    print(f"{datetime.now()}: avg_before: {avg_before}\navg_after: {avg_after}")
    print(f"{datetime.now()}: improve: {avg_before / avg_after}")

    book = openpyxl.Workbook()
    sheet_1 = book.create_sheet("results", 0)
    sheet_1.append(headers)

    for i in range(0, len(encodings_array_before)):
        sheet_1.append([i + 1, encodings_array_before[i], encodings_array_after[i]])
    book.save("results/results.xlsx")
