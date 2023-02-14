import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt

MODEL_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


class TransferModel:
    """
    a sample class to utilize tf hub style transfer model
    (transfers portraits to sketches quite okay,
    to perform transfer from sketches to portraits we should find something better)
    """

    def __init__(self, url=MODEL_URL):
        """
        initializes a model via tf hub url
        :param url: url to tensorflow hub model
        """
        self.model = hub.load(url)
        print(url)

    def transfer_style(self, content_img, style_img):
        """
        transfers a style from one image to another
        (images should be passed as tf tensors)
        :param content_img: image to be processed
        :param style_img: image which style to be transferred
        :return: input image with transferred style
        """
        res = self.model(tf.constant(content_img), tf.constant(style_img))[0]
        return self.__tensor_to_image(res)

    def process_image(self, content_path, style_path, result_path, should_display=False):
        """
        performs style transfer and displays the results if needed
        :param content_path: path to image to be processed
        :param style_path: path to image which style to be transferred
        :param result_path: path to save results to
        :param should_display: indicates whether images should be displayed
        :return: window with three images if prompted
        """
        content_img = self.__load_image(content_path)
        style_img = self.__load_image(style_path)
        result_img = self.transfer_style(content_img, style_img)
        imsave(result_path, result_img)
        if should_display:
            fig = plt.figure(figsize=(30, 5))
            fig.add_subplot(1, 3, 1)
            imshow(self.__tensor_to_image(content_img))
            fig.add_subplot(1, 3, 2)
            imshow(self.__tensor_to_image(style_img))
            fig.add_subplot(1, 3, 3)
            imshow(result_img)
            plt.show()

    @staticmethod
    def __tensor_to_image(tensor):
        """
        converts tf tensor to RGB uint8 image
        :param tensor: input tensor
        :return: numpy array with RGB uint8 image
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return tensor

    @staticmethod
    def __load_image(path):
        """
        reads image from given path
        :param path: path to image
        :return: tf tensor
        """
        max_dim = 512
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="style transfer script")
    parser.add_argument("-url", dest="url", default=None, type=str, help="url to tensorflow hub model")
    parser.add_argument("-ct", dest="content_path", required=True, type=str, help="path to content image")
    parser.add_argument("-st", dest="style_path", required=True, type=str, help="path to style image")
    parser.add_argument("-rs", dest="result_path", required=True, type=str, help="path to save the result image to")
    parser.add_argument("-d", dest="display", default=True, type=bool, help="prompt to display results")
    args = parser.parse_args()
    if args.url:
        tm = TransferModel(args.url)
    else:
        tm = TransferModel()
    tm.process_image(args.content_path, args.style_path, args.result_path, args.display)
