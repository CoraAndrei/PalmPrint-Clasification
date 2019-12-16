import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.io._io import imread
import os as _os


class GaborExtractFeatures(object):

    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats

    def match(feats, ref_feats):
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
        kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)

        return kernels

    def get_image(self):
        image = imread(_os.path.join("Histogram_processed/File_0.bmp"), as_gray=True)
        shrink = (slice(0, None, 3), slice(0, None, 3))
        hand = img_as_float(image)[shrink]
        # camera = img_as_float(data.camera())[shrink]
        image_names = ('hand')
        images = ([hand])

        return image, images, image_names

    def configure_reference_features(self):
        # prepare reference features
        ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
        ref_feats[0, :, :] = compute_feats(hand, kernels)
        ref_feats[1, :, :] = compute_feats(hand, kernels)

    def print_labels(self, kernels, ref_feats, picture_hand):
        print('Rotated images matched against references using Gabor filter banks:')

        print('original: imagine, rotated: 30deg, match result: ', 'end=')
        feats = compute_feats(ndi.rotate(picture_hand, angle=190, reshape=False), kernels)
        print(image_names[match(feats, ref_feats)])

        print('original: imagine, rotated: 70deg, match result: ', 'end=')
        feats = compute_feats(ndi.rotate(picture_hand, angle=70, reshape=False), kernels)
        print(image_names[match(feats, ref_feats)])

        print('original: imagine, rotated: 145deg, match result: ','end=')
        feats = compute_feats(ndi.rotate(picture_hand, angle=145, reshape=False), kernels)
        print(image_names[match(feats, ref_feats)])

    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

    def plot(self, power, images, image_names):
        # Plot a selection of the filter bank kernels and their responses.
        results = []
        kernel_params = []
        for theta in (0, 1):
            theta = theta / 4. * np.pi
            for frequency in (0.1, 0.4):
                kernel = gabor_kernel(frequency, theta=theta)
                params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
                kernel_params.append(params)
                # Save kernel and the power image for each image
                results.append((kernel, [power(img, kernel) for img in images]))
                print results[0][1]

        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
        plt.gray()

        fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

        axes[0][0].axis('off')

        # Plot original images
        for label, img, ax in zip(image_names, images, axes[0][1:]):
            ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis('off')

        for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
            # Plot Gabor kernel
            ax = ax_row[0]
            ax.imshow(np.real(kernel))
            ax.set_ylabel(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Gabor responses with the contrast normalized for each filter
            vmin = np.min(powers)
            vmax = np.max(powers)
            for patch, ax in zip(powers, ax_row[1:]):
                ax.imshow(patch, vmin=vmin, vmax=vmax)
                ax.axis('off')

        plt.show()

    def main_function(self):
        image, images, image_names = self.get_image()
        kernels = self.configure_kernels()
        feats = self.compute_feats(image, kernels)