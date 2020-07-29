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
        imsave(path, image)
