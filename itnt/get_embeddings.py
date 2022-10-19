import os
from datetime import datetime

import mkl_random
import numpy as np
from PIL import Image
from arcface.lib import ArcFaceModel

from utils.my_arcface.main import calculate_embedding_with_model
from utils.tensorflow.face_encoding import get_encoding

if __name__ == "__main__":

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

            # tensorflow embeddings
            portrait_image_embed_tf = get_encoding("portrait_resized.png")
            sketch_image_embed_tf = get_encoding('sketch_resized.png')
            random_sketch_image_embed_tf = get_encoding('random_sketch_resized.png')

            # arcface embeddings
            portrait_image_embed_arc = calculate_embedding_with_model("portrait_resized.png", input_size, model)
            sketch_image_embed_arc = calculate_embedding_with_model('sketch_resized.png', input_size, model)
            random_sketch_image_embed_arc = calculate_embedding_with_model('random_sketch_resized.png', input_size,
                                                                           model)

            if count < test_index:
                # ts
                count_to_save = count
                folder_name = "training"
            else:
                count_to_save = count - test_index
                folder_name = "test"

            path_to_save_right_portrait = os.path.join("../itnt", f"for_tf/{folder_name}", "right",
                                                       f"tf_portrait_embed_{count_to_save}")
            path_to_save_right_sketch = os.path.join("../itnt", f"for_tf/{folder_name}", "right",
                                                     f"tf_sketch_embed_{count_to_save}")
            path_to_save_wrong_portrait = os.path.join("../itnt", f"for_tf/{folder_name}", "wrong",
                                                       f"tf_portrait_embed_{count_to_save}")
            path_to_save_wrong_sketch = os.path.join("../itnt", f"for_tf/{folder_name}", "wrong",
                                                     f"tf_sketch_embed_{count_to_save}")
            np.save(path_to_save_right_portrait, portrait_image_embed_tf)
            np.save(path_to_save_right_sketch, sketch_image_embed_tf)
            np.save(path_to_save_wrong_portrait, portrait_image_embed_tf)
            np.save(path_to_save_wrong_sketch, random_sketch_image_embed_tf)

            # arc
            path_to_save_right_portrait = os.path.join("../itnt", f"for_arc/{folder_name}", "right",
                                                       f"arc_portrait_embed_{count_to_save}")
            path_to_save_right_sketch = os.path.join("../itnt", f"for_arc/{folder_name}", "right",
                                                     f"arc_sketch_embed_{count_to_save}")
            path_to_save_wrong_portrait = os.path.join("../itnt", f"for_arc/{folder_name}", "wrong",
                                                       f"arc_portrait_embed_{count_to_save}")
            path_to_save_wrong_sketch = os.path.join("../itnt", f"for_arc/{folder_name}", "wrong",
                                                     f"arc_sketch_embed_{count_to_save}")
            np.save(path_to_save_right_portrait, portrait_image_embed_arc)
            np.save(path_to_save_right_sketch, sketch_image_embed_arc)
            np.save(path_to_save_wrong_portrait, portrait_image_embed_arc)
            np.save(path_to_save_wrong_sketch, random_sketch_image_embed_arc)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue

        count += 1
