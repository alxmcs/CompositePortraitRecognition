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


def save_embeds_to_db(paths_vector, connection, model_name, dataset_name):
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
    cnt = 1
    date_added = datetime.datetime.now()
    count_errors = 0
    true_stylized_photo_errors = []
    false_stylized_photo_errors = []
    for paths in paths_vector:
        try:
            print(f'iteration number {cnt} from {len(paths_vector)}')

            # ============================ paths =================================#
            photo_with_true_style_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                                      str(cnt),
                                                      f'TD_CS_with_style_{cnt}.jpg')
            photo_with_false_style_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                                      str(cnt),
                                                      f'TD_CS_with_random_style_{cnt}.jpg')

            photo_with_true_style_path1 = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                                      str(cnt),
                                                      'true_stylized.jpg')
            photo_with_false_style_path1 = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                                       str(cnt),
                                                       'false_stylized.jpg')

            gray_photo_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                      str(cnt),
                                      f'TD_CS_image_grey_{cnt}.jpg')
            gray_true_sketch_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                            str(cnt),
                                            f'TD_CS_true_sketch_{cnt}.jpg')
            gray_false_sketch_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS',
                                             str(cnt),
                                             f'TD_CS_false_sketch_{cnt}.jpg')
            # ====================================================================== #

            last_id = insert_person(cursor, cnt, date_added)

            # ============================ embeddings ============================== #
            photo_embedding = get_embedding(paths[0], model_name)
            true_sketch_embedding = get_embedding(paths[1], model_name)
            false_sketch_embedding = get_embedding(paths[2], model_name)

            gray_photo_embedding = get_embedding(gray_photo_path, model_name)
            gray_true_sketch_embedding = get_embedding(gray_true_sketch_path, model_name)
            gray_false_sketch_embedding = get_embedding(gray_false_sketch_path, model_name)


            try:
                photo_with_true_style_embed = get_embedding(photo_with_true_style_path, model_name)
                photo_with_false_style_path_embed = get_embedding(photo_with_false_style_path, model_name)
                photo_with_true_style_embed1 = get_embedding(photo_with_true_style_path1, model_name)
            except ValueError:
                true_stylized_photo_errors.append(cnt)
                print(f'it:{cnt} true st error')
                try:
                    get_embedding(photo_with_false_style_path1, model_name)
                except ValueError:
                    false_stylized_photo_errors.append(cnt)
                    print(f'it:{cnt} false st error')
                cnt += 1
                continue

            try:
                photo_with_false_style_embed1 = get_embedding(photo_with_false_style_path1, model_name)
            except ValueError:
                false_stylized_photo_errors.append(cnt)
                cnt += 1
                continue
            # initial images
            insert_embedding(cursor, photo_embedding, date_added, model_id, last_id, None,
                             f'photo_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, true_sketch_embedding, date_added, model_id, last_id, None,
                             f'sketch_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, false_sketch_embedding, date_added, model_id, last_id, None,
                             f'sketch_false_{model_name}_{dataset_name}')
            # gray images
            insert_embedding(cursor, gray_photo_embedding, date_added, model_id, last_id, None,
                             f'gray_photo_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, gray_true_sketch_embedding, date_added, model_id, last_id, None,
                             f'gray_sketch_true_{model_name}_{dataset_name}')
            insert_embedding(cursor, gray_false_sketch_embedding, date_added, model_id, last_id, None,
                             f'gray_sketch_false_{model_name}_{dataset_name}')

            # stylized images(1 model)
            insert_embedding(cursor, photo_with_true_style_embed, date_added, model_id, last_id, None,
                             f'photo_true_{model_name}_{dataset_name}_st')
            insert_embedding(cursor, photo_with_false_style_path_embed, date_added, model_id, last_id, None,
                             f'photo_false_{model_name}_{dataset_name}_st')

            # stylized images(2 model)
            insert_embedding(cursor, photo_with_true_style_embed1, date_added, model_id, last_id, None,
                             f'photo_true_{model_name}_{dataset_name}_st1')
            insert_embedding(cursor, photo_with_false_style_embed1, date_added, model_id, last_id, None,
                             f'photo_false_{model_name}_{dataset_name}_st1')
            cnt += 1
        except IndexError as e:
            print(
                f"{str(e)} \n не удалось обнаружить лицо на фотографии")
            cnt += 1
            count_errors += 1
            continue
        except ValueError as e:
            print(e)
            cnt += 1
            count_errors += 1
            continue
        except Exception as e:
            print("unknown error")
            cnt += 1
            count_errors += 1
    print(true_stylized_photo_errors)
    print(false_stylized_photo_errors)
    connection.commit()
    print('The operation was successful')
    print(f'count errors = {count_errors}')


if __name__ == "__main__":

    db_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'db', 'database.db')
    with sqlite3.connect(db_path) as conn:
        dir_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'TDCS')
        tdcs_paths = common.dataset.TDCS.get_paths.get_paths(dir_path)
        print(tdcs_paths)
        dataset_name = 'tdcs'
        for i in range(0, len(models)):
            model_name = models[i]
            save_embeds_to_db(tdcs_paths, conn, model_name, dataset_name)
