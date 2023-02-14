import os
import sqlite3
import datetime
from deepface import DeepFace
import dlib_tf_project.utils.tensorflow.style_transfer
import common.dataset.TDCS.get_paths
import common.dataset.CUHK.get_paths
from common.db import db_operations
from common.db.db_operations import insert_person, insert_embedding

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
]


def insert_person(cursor, cnt, date_added):
    name = f'name_{cnt}'
    patronymic = f'patronymic_{cnt}'
    surname = f'surname_{cnt}'
    comment = ''
    last_id_int = db_operations.insert_person(cursor, name, patronymic, surname, comment, date_added)
    return last_id_int


def get_embedding(path, model_name):
    embedding_objs = DeepFace.represent(img_path=path,
                                        model_name=model_name
                                        )
    return embedding_objs


def save_embeds_to_db(paths_vector, connection, model_name, transfer_model, dataset_name):
    """

    :param paths_vector: vector [['photo_path', 'sketch_path', 'random_sketch_path'], ...];
    :param connection: database connection
    :param model_name: name of model to getting embeddings
    :param transfer_model: style transfer model
    :param dataset_name: name of dataset ("TDCS" or "CUHK")
    :return: nothing
    """

    cursor = connection.cursor()
    model_id = (cursor.execute("select id from model where name = ?", [model_name]).fetchone())[0]
    cnt = 0
    photo_with_true_style_path = os.path.join('image_with_true_style.png')
    photo_with_false_style_path = os.path.join('image_with_false_style.png')
    date_added = datetime.datetime.now()
    for paths in paths_vector:
        try:
            print(f'iteration number {cnt} from {len(paths_vector)}')
            photo_embedding = get_embedding(paths[0], model_name)
            true_sketch_embedding = get_embedding(paths[1], model_name)
            false_sketch_embedding = get_embedding(paths[2], model_name)

            transfer_model.process_image(paths[0], paths[1],
                                         photo_with_true_style_path)
            transfer_model.process_image(paths[0], paths[2],
                                         photo_with_false_style_path)
            photo_with_true_style_embed = get_embedding(photo_with_true_style_path, model_name)
            photo_with_false_style_path_embed = get_embedding(photo_with_false_style_path, model_name)

            last_id = insert_person(cursor, cnt, date_added)

            # without ST
            insert_embedding(cursor, photo_embedding, date_added, model_id, last_id, None,
                             f'photo_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, true_sketch_embedding, date_added, model_id, last_id, None,
                             f'sketch_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, false_sketch_embedding, date_added, model_id, last_id, None,
                             f'sketch_false_{model_name}_{dataset_name}')
            # with ST
            insert_embedding(cursor, photo_with_true_style_embed, date_added, model_id, last_id, None,
                             f'photo_true_{model_name}_{dataset_name}_st')
            insert_embedding(cursor, photo_with_false_style_path_embed, date_added, model_id, last_id, None,
                             f'photo_false_{model_name}_{dataset_name}_st')
            cnt += 1

        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")
            cnt += 1
            continue
        except ValueError as e:
            print(e)
            cnt += 1
            continue
    connection.commit()
    connection.close()
    print('The operation was successful')


if __name__ == "__main__":

    try:
        transfer_model = dlib_tf_project.utils.tensorflow.style_transfer.TransferModel(
            dlib_tf_project.utils.tensorflow.style_transfer.MODEL_URL)
    except (RuntimeError, TypeError, NameError):
        print('Error getting transfer model')
        exit()

    db_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'db', 'database.db')
    with sqlite3.connect(db_path) as conn:
        dir_path = os.path.join('/', 'common', 'dataset', 'TDCS')
        tdcs_paths = common.dataset.TDCS.get_paths.get_paths(dir_path)
        print(tdcs_paths)
        dataset_name = 'tdcs'
        for i in range(0, len(models)):
            model_name = models[i]
            save_embeds_to_db(tdcs_paths, conn, model_name, transfer_model, dataset_name)
