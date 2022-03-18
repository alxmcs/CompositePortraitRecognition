import utils.face_encoding
import utils.style_transfer

if __name__ == "__main__":

    transfer_model = utils.style_transfer.TransferModel(utils.style_transfer.MODEL_URL)

    images = ["images\photos\photo1.png"]

    sketches = ["images\sketches\sketch1.png"]

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
        # print("iteration number " + i.__str__())

    avg_before = sum(encodings_array_before)/len(encodings_array_before)
    avg_after = sum(encodings_array_after)/len(encodings_array_after)

    print("avg_before " + avg_before.__str__() + "\n" + "avg_after " + avg_after.__str__())

    # насколько стало лучше
    improve = avg_before/avg_after
    print("improve = " + improve.__str__())

    # если добавить 3 и 4 фотографии, то перенос стиля наоборот увеличит меру близости энкодингов изображений
