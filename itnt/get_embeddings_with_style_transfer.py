import os
from datetime import datetime

import mkl_random
import numpy as np
from PIL import Image
from arcface.lib import ArcFaceModel

import utils.tensorflow.style_transfer
from utils.my_arcface.main import calculate_embedding_with_model
from utils.tensorflow.face_encoding import get_encoding

if __name__ == "__main__":
    transfer_model = utils.tensorflow.style_transfer.TransferModel(
        utils.tensorflow.style_transfer.MODEL_URL)

    input_size = 300
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)
    size = 113

    count = 0
    test_index = int(size * 0.7)
    for i in range(1, size):
        print(f"{datetime.now()}: iteration number {i}")
        path1 = os.path.join("../dataset", "TDCS", str(i), f"TD_CS_{str(i)}.jpg")
        path0 = os.path.join("../dataset", "TDCS", str(i), "TD_RGB_E_1.jpg")
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

            portrait_image_with_style_right = os.path.join("../itnt", "image_with_style_right.png")
            portrait_image_with_style_wrong = os.path.join("../itnt", "image_with_style_wrong.png")

            image_with_style = transfer_model.process_image("portrait_resized.png", "sketch_resized.png",
                                                            portrait_image_with_style_right)
            image_with_random_style = transfer_model.process_image("portrait_resized.png", "random_sketch_resized.png",
                                                                   portrait_image_with_style_wrong)

            # tensorflow embeddings
            portrait_image_with_style_right_embed_tf = get_encoding(portrait_image_with_style_right)
            portrait_image_with_style_wrong_embed_tf = get_encoding(portrait_image_with_style_wrong)
            sketch_image_embed_right_tf = get_encoding('sketch_resized.png')
            random_sketch_image_embed_tf = get_encoding('random_sketch_resized.png')

            right_embed_tf = np.concatenate((portrait_image_with_style_right_embed_tf, sketch_image_embed_right_tf))
            wrong_embed_tf = np.concatenate((portrait_image_with_style_wrong_embed_tf, random_sketch_image_embed_tf))

            # arcface embeddings
            portrait_image_with_style_right_embed_arc = calculate_embedding_with_model(portrait_image_with_style_right,
                                                                                       input_size, model)
            portrait_image_with_style_wrong_embed_arc = calculate_embedding_with_model(portrait_image_with_style_wrong,
                                                                                       input_size, model)
            sketch_image_embed_right_arc = calculate_embedding_with_model('sketch_resized.png', input_size, model)
            random_sketch_image_embed_arc = calculate_embedding_with_model('random_sketch_resized.png', input_size,
                                                                           model)

            right_embed_arc = np.concatenate((portrait_image_with_style_right_embed_arc, sketch_image_embed_right_arc))
            wrong_embed_arc = np.concatenate((portrait_image_with_style_wrong_embed_arc, random_sketch_image_embed_arc))

            if count < test_index:
                # ts
                count_to_save = count
                folder_name = "training"
            else:
                count_to_save = count - test_index
                folder_name = "test"

            path_to_save_right = os.path.join("../itnt", f"for_tf_with_st/{folder_name}", "right",
                                              f"tf_embed_{count_to_save}")

            path_to_save_wrong = os.path.join("../itnt", f"for_tf_with_st/{folder_name}", "wrong",
                                              f"tf_embed_{count_to_save}")
            np.save(path_to_save_right, right_embed_tf)
            np.save(path_to_save_wrong, wrong_embed_tf)

            # arc
            path_to_save_right = os.path.join("../itnt", f"for_arc_with_st/{folder_name}", "right",
                                              f"arc_embed_{count_to_save}")

            path_to_save_wrong = os.path.join("../itnt", f"for_arc_with_st/{folder_name}", "wrong",
                                              f"arc_embed_{count_to_save}")

            np.save(path_to_save_right, right_embed_arc)
            np.save(path_to_save_wrong, wrong_embed_arc)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue

        count += 1
