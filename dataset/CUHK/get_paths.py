import os
import random

import numpy as np



def get_paths():
    # хард код, но я не знаю как по-другому ибо эта функция запускается из другой папки
    cur_dir = 'C:\\CompositePortraitRecongnition\\dataset\\CUHK'
    photos_path = os.path.join(cur_dir, 'photo')
    sketch_path = os.path.join(cur_dir, 'sketch')
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
    result = get_paths()
    print(result)
