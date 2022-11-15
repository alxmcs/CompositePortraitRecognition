import os
import random

import numpy as np

PHOTOS_PATH = 'C:\\CompositePortraitRecongnition\\dataset\\CUHK\\photo'
SKETCHES_PATH = 'C:\\CompositePortraitRecongnition\\dataset\\CUHK\\sketch'

def get_paths():
    photos_names = np.array(os.listdir(PHOTOS_PATH))
    len_dataset = len(photos_names)
    result = []
    for i in range(0, len_dataset):
        random_sketch_index = random.randint(0, len_dataset - 1)
        random_sketch_path = os.path.join(SKETCHES_PATH, f'{random_sketch_index}.jpg')
        result.append([os.path.join(PHOTOS_PATH, f'{i}.jpg'), os.path.join(SKETCHES_PATH, f'{i}.jpg'),
                       random_sketch_path])
    return result


if __name__ == "__main__":
    result = get_paths()
    print(result)
