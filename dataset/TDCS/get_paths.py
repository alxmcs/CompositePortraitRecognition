import os
import random


def get_paths():
    size = 113
    result = []
    # хард код, но я не знаю как по-другому ибо эта функция запускается из другой папки
    cur_dir = os.path.join('C:\\CompositePortraitRecongnition\\dataset\\TDCS')
    for i in range(1, size):
        photo_path = os.path.join(cur_dir, str(i), 'TD_RGB_E_1.jpg')
        photo_path = os.path.abspath(photo_path)
        sketch_path = os.path.join(cur_dir, str(i), f'TD_CS_{i}.jpg')
        sketch_path = os.path.abspath(sketch_path)
        rand_index = random.randint(1, size)
        random_sketch_path = os.path.join(cur_dir, str(rand_index), f'TD_CS_{rand_index}.jpg')
        random_sketch_path = os.path.abspath(random_sketch_path)
        result.append([photo_path, sketch_path, random_sketch_path])
    return result


if __name__ == "__main__":
    result = get_paths()
    print(result)
