import datetime
import os
import sqlite3

import numpy as np
from PIL import Image
from deepface.basemodels import ArcFace
from scipy.spatial import distance

from common.db.db_operations import insert_person, insert_embedding
from dlib_tf_project.utils.my_arcface.main import calculate_embedding_with_model
from dlib_tf_project.utils.tensorflow.face_encoding import get_encoding
import dlib_tf_project.utils.tensorflow.style_transfer
import common.dataset
import dlib_tf_project


def get_distances(u, v):
    e_d = distance.euclidean(u, v)
    ch_d = distance.chebyshev(u, v)
    cos = distance.cosine(u, v)
    return np.array([e_d, ch_d, cos])


def get_embeddings_for_paths(paths_vector, connection, arcface_model, arcface_input_size, transfer_model, dataset_name):
    """
    calculates all embedding and enters them into the database
    :param paths_vector: vector [['photo_path', 'sketch_path', 'random_sketch_path'], ...];
    :param connection: database connection
    :return:
    """

    cursor = connection.cursor()
    tensorflow_id = (cursor.execute("select id from model where name = ?", ['tensorflow']).fetchone())[0]
    arcface_id = (cursor.execute("select id from model where name = ?", ['arcface']).fetchone())[0]

    thumbnail_id = (cursor.execute("select id from preprocessing where name = ?", ['thumbnail']).fetchone())[0]
    cnt = 0
    comment = ''
    for paths in paths_vector:
        try:
            print(f'iteration number {cnt} from {len(paths_vector)}')

            portrait_image = Image.open(paths[0])
            portrait_image.thumbnail((arcface_input_size, arcface_input_size))
            portrait_image.save("portrait_resized.png")

            sketch_image = Image.open(paths[1])
            sketch_image.thumbnail((arcface_input_size, arcface_input_size))
            sketch_image.save('sketch_resized.png')

            random_sketch_image = Image.open(paths[2])
            random_sketch_image.thumbnail((arcface_input_size, arcface_input_size))
            random_sketch_image.save('random_sketch_resized.png')

            portrait_image_with_style_right = os.path.join('image_with_style_right.png')
            portrait_image_with_style_wrong = os.path.join('image_with_style_wrong.png')

            transfer_model.process_image("portrait_resized.png", "sketch_resized.png",
                                         portrait_image_with_style_right)
            transfer_model.process_image("portrait_resized.png", "random_sketch_resized.png",
                                         portrait_image_with_style_wrong)

            image_with_style_embed_tf = get_encoding(portrait_image_with_style_right)
            image_with_random_style_embed_tf = get_encoding(portrait_image_with_style_wrong)

            image_with_style_embed_arc = calculate_embedding_with_model(portrait_image_with_style_right,
                                                                        arcface_input_size, arcface_model)
            image_with_random_style_embed_arc = calculate_embedding_with_model(portrait_image_with_style_wrong,
                                                                               arcface_input_size, arcface_model)

            # tensorflow embeddings
            portrait_image_embed_tf = get_encoding("portrait_resized.png")
            sketch_image_embed_tf = get_encoding('sketch_resized.png')
            random_sketch_image_embed_tf = get_encoding('random_sketch_resized.png')

            # arcface embeddings
            portrait_image_embed_arc = calculate_embedding_with_model("portrait_resized.png", arcface_input_size,
                                                                      arcface_model)
            sketch_image_embed_arc = calculate_embedding_with_model('sketch_resized.png', arcface_input_size,
                                                                    arcface_model)
            random_sketch_image_embed_arc = calculate_embedding_with_model('random_sketch_resized.png',
                                                                           arcface_input_size,
                                                                           arcface_model)

            name = f'name_{cnt}'
            patronymic = f'patronymic_{cnt}'
            surname = f'surname_{cnt}'

            date_added = datetime.datetime.now()

            last_id_int = insert_person(cursor, name, patronymic, surname, comment, date_added)

            insert_embedding(cursor, portrait_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
                             f'photo_true_tf_{dataset_name}')
            insert_embedding(cursor, sketch_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
                             f'sketch_true_tf_{dataset_name}')
            insert_embedding(cursor, random_sketch_image_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
                             f'sketch_false_tf_{dataset_name}')

            insert_embedding(cursor, portrait_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
                             f'photo_true_arc_{dataset_name}')
            insert_embedding(cursor, sketch_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
                             f'sketch_true_arc_{dataset_name}')
            insert_embedding(cursor, random_sketch_image_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
                             f'sketch_false_arc_{dataset_name}')

            insert_embedding(cursor, image_with_style_embed_tf, date_added, tensorflow_id, last_id_int, thumbnail_id,
                             f'image_with_st_tf_{dataset_name}')

            insert_embedding(cursor, image_with_style_embed_arc, date_added, arcface_id, last_id_int, thumbnail_id,
                             f'image_with_st_arc_{dataset_name}')

            insert_embedding(cursor, image_with_random_style_embed_tf, date_added, tensorflow_id, last_id_int,
                             thumbnail_id,
                             f'image_with_random_st_tf_{dataset_name}')

            insert_embedding(cursor, image_with_random_style_embed_arc, date_added, arcface_id, last_id_int,
                             thumbnail_id,
                             f'image_with_random_st_arc_{dataset_name}')


        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")  # https://stackoverflow.com/questions/59919993/indexerror-list-index-out-of-range-face-recognition
            continue
        cnt += 1


if __name__ == "__main__":
    model = ArcFace.loadModel()
    # model.load_weights("arcface_weights.h5")

    transfer_model = dlib_tf_project.utils.tensorflow.style_transfer.TransferModel(
        dlib_tf_project.utils.tensorflow.style_transfer.MODEL_URL)

    db_path = os.path.join('C:\\CompositePortraitRecongnition', 'db', 'database.db')
    connection = sqlite3.connect(db_path)

    dir_path = os.path.join('C:\\CompositePortraitRecongnition', 'dataset', 'TDCS')
    tdcs_paths = common.dataset.TDCS.get_paths.get_paths(dir_path)
    print(tdcs_paths)
    input_size = 112
    get_embeddings_for_paths(tdcs_paths, connection, model, input_size, transfer_model, 'tdcs')
    connection.commit()
    connection.close()
