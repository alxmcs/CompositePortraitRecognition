import datetime
import os
import sqlite3
from scipy.spatial import distance
import mkl_random
import numpy as np
from PIL import Image
from arcface.lib import ArcFaceModel

from db.db_operations import insert_person, insert_embedding
from utils.my_arcface.main import calculate_embedding_with_model
from utils.tensorflow.face_encoding import get_encoding


def get_distances(u, v):
    e_d = distance.euclidean(u, v)
    ch_d = distance.chebyshev(u, v)
    cos = distance.cosine(u, v)
    return np.array([e_d, ch_d, cos])


if __name__ == "__main__":

    input_size = 300
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)

    # path_dataset = os.path.join('..//dataset', 'TDCS')
    # folder_counter = sum([len(folder) for r, d, folder in os.walk(path_dataset)])
    # size = folder_counter

    size = 113

    # conn = sqlite3.connect("C:\\CompositePortraitRecongnition\\db\\database.db")
    # cursor = conn.cursor()
    #
    # tensorflow_id = (cursor.execute("select id from model where name = ?", ['tensorflow']).fetchone())[0]
    # arcface_id = (cursor.execute("select id from model where name = ?", ['arcface']).fetchone())[0]
    #
    # thumbnail_id = (cursor.execute("select id from preprocessing where name = ?", ['thumbnail']).fetchone())[0]

    count = 0
    test_index = int(size * 0.7)

    comment = 'for testing tf and arcface'

    for i in range(1, size):
        print(f"{datetime.datetime.now()}: iteration number {i}")
        # sketch
        path1 = os.path.join("../../dataset", "TDCS", str(i), f"TD_CS_{str(i)}.jpg") \
            # photo
        path0 = os.path.join("../../dataset", "TDCS", str(i), "TD_RGB_E_1.jpg")
        random_sketch_index = mkl_random.randint(1, 113)
        # random sketch
        path2 = os.path.join("../../dataset", "TDCS", str(random_sketch_index), f"TD_CS_{str(random_sketch_index)}.jpg")
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
            portrait_image_embed_tf = get_encoding("../portrait_resized.png")
            sketch_image_embed_tf = get_encoding('../sketch_resized.png')
            random_sketch_image_embed_tf = get_encoding('../random_sketch_resized.png')

            right_distances_tf = get_distances(portrait_image_embed_tf, sketch_image_embed_tf)
            wrong_distances_tf = get_distances(portrait_image_embed_tf, random_sketch_image_embed_tf)

            right_embed_tf = np.concatenate((portrait_image_embed_tf, sketch_image_embed_tf))
            wrong_embed_tf = np.concatenate((portrait_image_embed_tf, random_sketch_image_embed_tf))

            # arcface embeddings
            portrait_image_embed_arc = calculate_embedding_with_model("../portrait_resized.png", input_size, model)
            sketch_image_embed_arc = calculate_embedding_with_model('../sketch_resized.png', input_size, model)
            random_sketch_image_embed_arc = calculate_embedding_with_model('../random_sketch_resized.png', input_size,
                                                                           model)
            right_distances_arc = get_distances(portrait_image_embed_arc, sketch_image_embed_arc)
            wrong_distances_arc = get_distances(portrait_image_embed_arc, random_sketch_image_embed_arc)

            # name = f'name_{i}'
            # patronymic = f'patronymic_{i}'
            # surname = f'surname_{i}'
            #
            # date_added = datetime.datetime.now()
            #
            # last_id_int = insert_person(cursor, name, patronymic, surname, comment, date_added)
            #
            # insert_embedding(cursor, portrait_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
            #                  'photo_true_tf')
            # insert_embedding(cursor, sketch_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
            #                  'sketch_true_tf')
            # insert_embedding(cursor, random_sketch_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
            #                  'sketch_false_tf')
            #
            # insert_embedding(cursor, portrait_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
            #                  'photo_true_arc')
            # insert_embedding(cursor, sketch_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
            #                  'sketch_true_arc')
            # insert_embedding(cursor, random_sketch_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
            #                  'sketch_false_arc')

            right_embed_arc = np.concatenate((portrait_image_embed_arc, sketch_image_embed_arc))
            wrong_embed_arc = np.concatenate((portrait_image_embed_arc, random_sketch_image_embed_arc))

            # запись эмбедингов(конкатенации) в файл
            if count < test_index:
                # ts
                count_to_save = count
                folder_name = "training"
            else:
                count_to_save = count - test_index
                folder_name = "test"

            path_to_save_right = os.path.join("..", "TDCS", f"for_tf/{folder_name}", "right",
                                              f"tf_embed_{count_to_save}")
            path_to_save_right_distances = os.path.join("..", "TDCS", f"for_tf/{folder_name}", "right",
                                                        f"tf_embed_distances_{count_to_save}")
            path_to_save_wrong = os.path.join("..", "TDCS", f"for_tf/{folder_name}", "wrong",
                                              f"tf_embed_{count_to_save}")
            path_to_save_wrong_distances = os.path.join("..", "TDCS", f"for_tf/{folder_name}", "wrong",
                                                        f"tf_embed_distances_{count_to_save}")

            np.save(path_to_save_right, right_embed_tf)
            np.save(path_to_save_wrong, wrong_embed_tf)
            np.save(path_to_save_right_distances, right_distances_tf)
            np.save(path_to_save_wrong_distances, wrong_distances_tf)

            # arc
            path_to_save_right = os.path.join("..", "TDCS", f"for_arc/{folder_name}", "right",
                                              f"arc_embed_{count_to_save}")

            path_to_save_wrong = os.path.join("..", "TDCS", f"for_arc/{folder_name}", "wrong",
                                              f"arc_embed_{count_to_save}")

            path_to_save_right_distances = os.path.join("..", "TDCS", f"for_arc/{folder_name}", "right",
                                                        f"arc_embed_distances_{count_to_save}")

            path_to_save_wrong_distances = os.path.join("..", "TDCS", f"for_arc/{folder_name}", "wrong",
                                                        f"arc_embed_distances_{count_to_save}")

            np.save(path_to_save_right, right_embed_arc)
            np.save(path_to_save_wrong, wrong_embed_arc)
            np.save(path_to_save_right_distances, right_distances_arc)
            np.save(path_to_save_wrong_distances, wrong_distances_arc)
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue

        count += 1

    # conn.commit()
    # conn.close()
