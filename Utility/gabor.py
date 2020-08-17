import cv2
import glob
import numpy as np

from skimage.io import imread_collection
from scipy import ndimage as ndi

from skimage.util import img_as_float


class GaborExtractFeatures(object):

    def get_image(self, processed_image):
        shrink = (slice(0, None, 3), slice(0, None, 3))
        hand = img_as_float(processed_image)[shrink]
        image_names = 'hand'
        images = ([hand])
        return processed_image, images, image_names, hand

    def power(self, image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    def plot(self, images, class_name, index, general_index, second=False, both=False):
        # Plot a selection of the filter bank kernels and their responses.
        results2 = []

        for theta in (1.4, 2.1, 3, 4):
            for ksize in (11, 15):
                sigma = 1
                gamma = 0.5
                lamda = 0.9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_64F)
                results2.append([self.power(img, kernel) for img in images])

        aaa = ""
        for i in results2:
            for a in i:
                np_array = np.array(a)
                numX = np_array.shape[1]
                covX = np.zeros((1))
                meanX = covX.mean()

                for k in range(0, numX):
                    covX = covX + (np_array[:, k] - meanX) * (np_array[:, k] - meanX)
                covX = covX / (numX - 1)

                for c in covX:
                    aaa += str(c) + ','

        features_length = len(aaa.split(','))

        if second is True and both is True:
            aaa += "img{}".format(str(class_name))
        elif both is False:
            aaa += "img{}".format(str(class_name))

        # Gabor Learn csv
        if index == 0:
            with open('Gabor_learn.csv', 'w') as fp:
                if both is True:
                    for column in range(features_length*2-2):
                        fp.write("attr_{},".format(column))
                else:
                    for column in range(features_length-1):
                        fp.write("attr_{},".format(column))
                fp.write("class_name")
                fp.write('\n')

        # Gabor Train csv
        if index == 0:
            with open('Gabor_test.csv', 'w') as fp:
                if both is True:
                    for column in range(features_length * 2 - 2):
                        fp.write("attr_{},".format(column))
                else:
                    for column in range(features_length - 1):
                        fp.write("attr_{},".format(column))
                fp.write("class_name")
                fp.write('\n')
        if both:
            if general_index in [1, 2, 3, 4, 5, 6, 7, 8]:
                with open('Gabor_learn.csv', 'a') as fp:
                    fp.write(aaa)
                    if second and both:
                        fp.write('\n')
                    elif both is False:
                        fp.write('\n')
            elif general_index in [9, 10]:
                with open('Gabor_test.csv', 'a') as fp:
                    fp.write(aaa)
                    if second and both:
                        fp.write('\n')
                    elif both is False:
                        fp.write('\n')
        else:
            if general_index in [1, 2, 3, 4]:
                with open('Gabor_learn.csv', 'a') as fp:
                    fp.write(aaa)
                    if second and both:
                        fp.write('\n')
                    elif both is False:
                        fp.write('\n')

            elif general_index in [5]:
                with open('Gabor_test.csv', 'a') as fp:
                    fp.write(aaa)
                    if second and both:
                        fp.write('\n')
                    elif both is False:
                        fp.write('\n')

    def gabor_plot(self, processed_images_path, both=False):
        base = glob.glob(processed_images_path)
        all_images = imread_collection(processed_images_path)
        count = 0
        for index, image in enumerate(all_images):
            class_name = base[index].split('\\')[-1].replace(".bmp", "")[0:3]
            class_double = base[index].split('\\')[-1].replace(".bmp", "")[3:6]
            self.second = True if 'a' in class_double else False
            image, images, image_names, hand = self.get_image(image)
            count += 1
            self.plot(images, class_name=class_name, index=index, general_index=count,
                      second=self.second, both=both)
            if both:
                if count == 10:
                    count = 0
            elif both is False:
                if count == 5:
                    count = 0
