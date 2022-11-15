import os
import random

import numpy as np


def get_paths():
    size = 113
    result = []
    for i in range(1, size):
        photos_path = os.path.join('C:\CompositePortraitRecongnition\dataset\TDCS', str(i), 'TD_RGB_E_1.jpg')
        sketch_path = os.path.join('C:\CompositePortraitRecongnition\dataset\TDCS', str(i), f'TD_CS_{i}.jpg')
        rand_index = random.randint(1, size)
        random_sketch_path = os.path.join('C:\CompositePortraitRecongnition\dataset\TDCS', str(rand_index), f'TD_CS_{rand_index}.jpg')
        result.append([photos_path, sketch_path, random_sketch_path])
    return result


if __name__ == "__main__":
    result = get_paths()
    print(result)
