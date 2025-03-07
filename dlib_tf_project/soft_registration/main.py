import json
import dlib_tf_project.utils.tensorflow.face_encoding
from PIL import Image
from arcface.lib import ArcFaceModel

import dlib_tf_project.utils.tensorflow.face_encoding


if __name__ == "__main__":

    with open('settings.json') as info_data:
        json_data = json.load(info_data)

    portrait_path = json_data['portrait_path']
    sketch_path = json_data['sketch_path']
    threshold = json_data['threshold']

    tensorflow_distance = dlib_tf_project.utils.tensorflow.face_encoding.calculate_distance(portrait_path, sketch_path)
    tensorflow_result = False
    if tensorflow_distance < threshold:
        tensorflow_result = True

    input_size = 300
    portrait_image = Image.open(portrait_path)
    print(f"Original size : {portrait_image.size}")  # 5464x3640

    portrait_image.thumbnail((input_size, input_size))
    portrait_image.save("portrait_resized.png")

    sketch_image = Image.open(sketch_path)
    print(f"Original size : {sketch_image.size}")  # 5464x3640

    sketch_image.thumbnail((input_size, input_size))
    sketch_image.save('sketch_resized.png')

    # arcface model
    model = ArcFaceModel(size=input_size,
                         backbone_type='ResNet50',
                         training=False)
    arcface_distance = dlib_tf_project.utils.my_arcface.main.calculate_distance('portrait_resized.png', 'sketch_resized.png',
                                                                input_size, model)
    arcface_result = False
    if arcface_distance < threshold:
        arcface_result = True
    results = {'tensorflow_distance': str(tensorflow_distance), 'tensorflow_is_like': tensorflow_result,
               'arcface_distance': str(arcface_distance), 'arcface_is_like': arcface_result}
    with open('results.json', 'w') as fp:
        json.dump(results, fp)
