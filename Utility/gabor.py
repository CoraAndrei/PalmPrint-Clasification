import cv2
import glob

import numpy
from skimage.io import imread_collection
import numpy as np
from scipy import ndimage as ndi

from sklearn import random_projection


from skimage.util import img_as_float, img_as_float64, img_as_uint
from skimage.filters import gabor_kernel


class GaborExtractFeatures(object):

    def compute_feats(self, image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats

    def match(self, feats, ref_feats):
        min_error = np.inf
        min_i = None
        for i in range(ref_feats.shape[0]):
            error = np.sum((feats - ref_feats[i, :])**2)
            if error < min_error:
                min_error = error
                min_i = i
        return min_i

    def configure_kernels(self):
        # prepare filter bank kernels
        # 16 arrays
        kernels = []

        theta = 2
        sigma = 1
        frequency = 0.05
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel.astype(float))
        # kernels = []
        # for theta in range(4):
        #     theta = theta / 4. * np.pi
        #     for sigma in (1, 3):
        #         for frequency in (0.05, 0.25):
        #             kernel = np.real(gabor_kernel(frequency, theta=theta,
        #                                           sigma_x=sigma, sigma_y=sigma))
        #             print ("kernelLLL: {}".format(kernel))
        #             kernels.append(kernel.astype(float))
        return kernels

    def get_image(self, processed_image):
        # image = imread(_os.path.join("Histogram_processed/File_0.bmp"), as_gray=True)
        #img = cv2.imread(processed_image)
        shrink = (slice(0, None, 3), slice(0, None, 3))
        hand = img_as_float(processed_image)[shrink]
        #hand = hand.reshape(-1)
#       hand = img_as_float(processed_image)[shrink]
        image_names = 'hand'
        images = ([hand])

        return processed_image, images, image_names, hand

    def configure_reference_features(self, hand, kernels):
        # prepare reference features
        ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
        ref_feats[0, :, :] = self.compute_feats(hand, kernels)
        ref_feats[1, :, :] = self.compute_feats(hand, kernels)
        return ref_feats

    def print_labels(self, kernels, ref_feats, hand, image_names):
        print('Rotated images matched against references using Gabor filter banks:')

        print('original: imagine, rotated: 30deg, match result: ', 'end=')
        feats = self.compute_feats(ndi.rotate(hand, angle=190, reshape=False), kernels)
        print(image_names[self.match(feats, ref_feats)])

        print('original: imagine, rotated: 70deg, match result: ', 'end=')
        feats = self.compute_feats(ndi.rotate(hand, angle=70, reshape=False), kernels)
        print(image_names[self.match(feats, ref_feats)])

        print('original: imagine, rotated: 145deg, match result: ', 'end=')
        feats = self.compute_feats(ndi.rotate(hand, angle=145, reshape=False), kernels)
        print(image_names[self.match(feats, ref_feats)])

    def power(self, image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    def plot(self, images, image_names, class_name, index, second=False):
        # Plot a selection of the filter bank kernels and their responses.
        results2 = []
        # for theta in (0, 1):  # 0 = 0 degrees, 1 = 45 degrees
        #     theta = theta / 4. * np.pi
        #     for frequency in (0.1, 0.4):  # frequency 0.10 and 0.40
        #         #theta = 2
        #         sigma = 1
        #         #frequency = 0.05
        #         kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
        #
        #         # Save kernel and the power image for each image
        #         results2.append([self.power(img, kernel) for img in images])

        for theta in (1.4, 2.1):
            for ksize in (11, 15):
                sigma = 1
                # ksize = 15
                gamma = 0.5
                lamda = 0.9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_64F)

            #kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                results2.append([self.power(img, kernel) for img in images])

        # np_array = np.array(results2[0][0])
        # array_mean = np_array.mean(axis=1)
        # print ('np_array : {}'.format(np_array))
        # print ('np_array_mean : {}'.format(array_mean))
        # print ('results2 : {}'.format(results2[0][0]))
                # se inmulteste valoarea kernel cu pixeli imagini?

        #print ('Damira: {}'.format(results2[0][0][0]))
        #print ('Damira: {} '.format('%.8f' %  (results2[0][0][0][0])))

        # kernels = []
        # for theta in range(2):
        #     theta = theta / 4. * np.pi
        #     for sigma in (3, 5):
        #         for lamda in np.arange(0, np.pi, np.pi / 4.):
        #             for gamma in (0.05, 0.5):
        #                 ksize = 9
        #                 kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_64F)
        #                 kernels.append(kernel)
        #                 results2 += str([cv2.filter2D(img, cv2.CV_8UC3, kernel) for img in images])

        # np_array = np.array(results2[0][0])
        # array_mean = np_array.mean(axis=1)
        # print ('np_array : {}'.format(np_array))
        # print ('np_array_mean : {}'.format(array_mean))
        # print ('results2 : {}'.format(results2[0][0]))

        aaa = ""
        for i in results2:
            for a in i:
                # for b in a:
                np_array = np.array(a)
                numX = np_array.shape[1]
                covX = numpy.zeros((1))
                meanX = covX.mean()

                # mean = np_array.mean(axis=1)
                for k in range(0, numX):
                    covX = covX + (np_array[:, k] - meanX) * (np_array[:, k] - meanX)
                covX = covX / (numX - 1)

                for c in covX:
                    aaa += str(c) + ','

        features_length = len(aaa.split(','))
        if second:
            aaa += "img{}".format(str(class_name))

        if index == 0:
            with open('Gabor_results.csv', 'w') as fp:
                for column in range(features_length*2-2):
                    fp.write("attr_{},".format(column))
                fp.write("class_name")
                fp.write('\n')
        with open('Gabor_results.csv', 'a') as fp:
            fp.write(aaa)
            if second:
                fp.write('\n')

        # fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 7))
        # plt.gray()
        #
        # fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
        #
        # axes[0][0].axis('off')
        #
        # # Plot original images
        # for label, img, ax in zip(image_names, images, axes[0][1:]):
        #     ax.imshow(img)
        #     ax.set_title(label, fontsize=9)
        #     ax.axis('off')
        #
        # for label, (kernel, powers), ax_row in zip(kernel_params, results2, axes[1:]):
        #     # Plot Gabor kernel
        #     ax = ax_row[0]
        #     ax.imshow(np.real(kernel))
        #     ax.set_ylabel(label, fontsize=7)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #
        #     # Plot Gabor responses with the contrast normalized for each filter
        #     vmin = np.min(powers)
        #     vmax = np.max(powers)
        #     for patch, ax in zip(powers, ax_row[1:]):
        #         ax.imshow(patch, vmin=vmin, vmax=vmax)
        #         ax.axis('off')

        #plt.show()

    def gabor_plot(self, processed_images_path):
        base = glob.glob(processed_images_path)
        all_images = imread_collection(processed_images_path)
        for index, image in enumerate(all_images):
            class_name = base[index].split('\\')[-1].replace(".bmp", "")[0:3]
            class_double = base[index].split('\\')[-1].replace(".bmp", "")[3:6]
            self.second = True if 'a' in class_double else False
            image, images, image_names, hand = self.get_image(image)
            #kernels = self.configure_kernels()
            #feats = self.compute_feats(image, kernels)
            #ref_feats = self.configure_reference_features(hand, kernels)

            #self.print_labels(kernels, ref_feats, hand, image_names)
            self.plot(images, image_names, class_name=class_name, index=index, second=self.second)
