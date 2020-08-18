import argparse

from skimage.io import imread_collection
from skimage.io._plugins.pil_plugin import imsave


class ImageLoader(object):
    def add_images_to_collection(self, path):
        """
        Method adds images from a certain path into a list
        path: Custom location where the images are located
        return: collection of images
        """
        # creating a collection with the available images
        col = imread_collection(path)
        return col

    def save_image(self, image, path):
        """ Method that saves the given image to a given path
        :param image: given image to be saved
        :param path: save location path
        """
        imsave(path, image)

    def str2bool(self, v):
        """ Method that is checking if keyboard input should return True or False
        :param v: keyboard input - argument
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
