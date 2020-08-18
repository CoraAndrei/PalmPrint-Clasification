import cv2
import glob
import numpy as np

from skimage.io import imread_collection
from scipy import ndimage as ndi

from skimage.util import img_as_float


class GaborExtractFeatures(object):

    def get_image(self, processed_image):
        """ Method that is shrinking the initial image and transforms it into a float array
        :param processed_image: image that is transformed into a float array
        """
        shrink = (slice(0, None, 3), slice(0, None, 3))
        hand_image = img_as_float(processed_image)[shrink]
        return hand_image

    def power(self, image, kernel):
        """ Method that is multiplying the values from image to the ones from the kernel matrix(real and imaginary)"""
        # Normalize image for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    def plot(self, image, class_name, index, general_index, train_test_ratio, second_class=False, both_images=False):
        """
        Method that is applying gabor filter on a image
        :param image: image on which Gabor filter is applied
        :param class_name: <str> image name that is set in csv file
        :param index: <int> only when index is 0 the csv file is populated with the number of arguments header
        :param general_index: <int> is used when adding a certain number of classes in test and train csv file
        :param second_class: <bool> if second is True, both hands are used and two classes will be concatenated
        :param both_images: <bool> if True, both hands are used for feature extraction
        :param train_test_ratio: <str> ration between train images and test images e.g.: 3/2 is 3 train images, 2 test
        """
        processed_values = []

        for theta in (1.4, 2.1, 3, 4):
            for ksize in (11, 15):
                sigma = 1
                gamma = 0.5
                lamda = 0.9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_64F)
                processed_values.append([self.power(image, kernel)])

        csv_results = ""
        for item in processed_values:
            for arr in item:
                np_array = np.array(arr)
                num_x = np_array.shape[1]
                cov_x = np.zeros(1)
                mean_x = cov_x.mean()

                for k in range(0, num_x):
                    cov_x = cov_x + (np_array[:, k] - mean_x) * (np_array[:, k] - mean_x)
                cov_x = cov_x / (num_x - 1)

                for result in cov_x:
                    csv_results += str(result) + ','

        features_length = len(csv_results.split(','))

        # adding image number to class img at the end of each line in csv
        if second_class is True and both_images is True:
            csv_results += "img{}".format(str(class_name))
        elif both_images is False:
            csv_results += "img{}".format(str(class_name))

        # Gabor Train csv - adding initial arguments line to the train csv file
        if index == 0:
            with open('Gabor_train.csv', 'w') as fp:
                if both_images is True:
                    for column in range(features_length*2-2):
                        fp.write("attr_{},".format(column))
                else:
                    for column in range(features_length-1):
                        fp.write("attr_{},".format(column))
                fp.write("class_name")
                fp.write('\n')

        # Gabor Train csv - adding initial arguments line to the train csv file
        if index == 0:
            with open('Gabor_test.csv', 'w') as fp:
                if both_images is True:
                    for column in range(features_length * 2 - 2):
                        fp.write("attr_{},".format(column))
                else:
                    for column in range(features_length - 1):
                        fp.write("attr_{},".format(column))
                fp.write("class_name")
                fp.write('\n')

        if both_images:
            if train_test_ratio == "3/2" or '3/2':
                self.train_index_both = [1, 2, 3, 4, 5, 6]
                self.test_index_both = [7, 8, 9, 10]
            elif train_test_ratio == "4/1" or '4/1':
                self.train_index_both = [1, 2, 3, 4, 5, 6, 7, 8]
                self.test_index_both = [9, 10]

            if general_index in self.train_index_both:
                with open('Gabor_train.csv', 'a') as fp:
                    fp.write(csv_results)
                    if second_class and both_images:
                        fp.write('\n')
                    elif both_images is False:
                        fp.write('\n')
            elif general_index in self.test_index_both:
                with open('Gabor_test.csv', 'a') as fp:
                    fp.write(csv_results)
                    if second_class and both_images:
                        fp.write('\n')
                    elif both_images is False:
                        fp.write('\n')
        else:
            if train_test_ratio == "3/2" or '3/2':
                self.train_index_single = [1, 2, 3]
                self.test_index_single = [4, 5]
            elif train_test_ratio == "4/1" or '4/1':
                self.train_index_single = [1, 2, 3, 4]
                self.test_index_single = [5]
            if general_index in self.train_index_single:
                with open('Gabor_train.csv', 'a') as fp:
                    fp.write(csv_results)
                    if second_class and both_images:
                        fp.write('\n')
                    elif both_images is False:
                        fp.write('\n')

            elif general_index in self.test_index_single:
                with open('Gabor_test.csv', 'a') as fp:
                    fp.write(csv_results)
                    if second_class and both_images:
                        fp.write('\n')
                    elif both_images is False:
                        fp.write('\n')

    def gabor_plot(self, processed_images_path, train_test_ratio, both_images=False):
        """ Method that is creating a collection of images from a given path of the processed images
        :param processed_images_path: path where the processed images are located
        :param both_images: if True, both hands are used (left, right)
        :param train_test_ratio: <str> ration between train images and test images e.g.: 3/2 is 3 train images, 2 test
        """
        base = glob.glob(processed_images_path)
        all_images = imread_collection(processed_images_path)
        count = 0
        for index, image in enumerate(all_images):
            class_name = base[index].split('\\')[-1].replace(".bmp", "")[0:3]    # 001, 002
            class_double = base[index].split('\\')[-1].replace(".bmp", "")[3:6]  # _1, _1a

            self.second_class = True if 'a' in class_double else False
            hand_image = self.get_image(image)
            count += 1
            self.plot(hand_image, class_name=class_name, index=index, general_index=count,
                      second_class=self.second_class, both_images=both_images, train_test_ratio=train_test_ratio)
            if both_images:
                if count == 10:
                    count = 0
            elif both_images is False:
                if count == 5:
                    count = 0
