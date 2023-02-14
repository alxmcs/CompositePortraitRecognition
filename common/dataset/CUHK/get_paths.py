import os
import random

import numpy as np


def get_paths(dir_path):
    photos_path = os.path.join(dir_path, 'photo')
    sketch_path = os.path.join(dir_path, 'sketch')
    photos_names = np.array(os.listdir(photos_path))
    len_dataset = len(photos_names)
    result = []
    for i in range(0, len_dataset):
        random_sketch_index = random.randint(0, len_dataset - 1)
        random_sketch_path = os.path.join(sketch_path, f'{random_sketch_index}.jpg')
        result.append([os.path.join(photos_path, f'{i}.jpg'), os.path.join(sketch_path, f'{i}.jpg'),
                       random_sketch_path])
    return result


if __name__ == "__main__":
    print(os.getcwd())
    cur_dir = os.path.join('/', 'dataset', 'CUHK')
    result = get_paths(cur_dir)
    print(result)
