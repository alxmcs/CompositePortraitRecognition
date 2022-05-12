import argparse
import face_recognition


def get_encoding(path):
    """
    reads image from a given location and calculates its face encoding
    (mage is considered to portrait a single person)
    :param path: path to the image file
    :return: array with 128 elements, containing a single face encoding
    """
    image = face_recognition.load_image_file(path)
    return face_recognition.api.face_encodings(image)[0]


def calculate_distance(path0, path1):
    """
    reads two images from a given location and calculates distance between their face encodings
    (each image is considered to portrait a single person)
    :param path0: path to the first image file
    :param path1:path to the second  file
    :return: euclidean distance between image encodings
    """
    enc0 = get_encoding(path0)
    enc1 = get_encoding(path1)
    return face_recognition.face_distance([enc0], enc1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face encoding script")
    parser.add_argument("-p0", dest="path0", required=True, type=str, help="path to the first image")
    parser.add_argument("-p1", dest="path1", required=True, type=str, help="path to the second image")
    args = parser.parse_args()
    print('calculating euclidean distance between two face encodings:')
    print(f'first: {args.path0}\nsecond: {args.path1}\nresult: {calculate_distance(args.path0, args.path1)}')
