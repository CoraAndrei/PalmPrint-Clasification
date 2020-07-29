import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io._io import imread
import os as _os

from skimage import img_as_float, filters
from skimage import exposure

matplotlib.rcParams['font.size'] = 8

class HistogramShow:
    def plot_img_and_hist(self, image, axes, bins=256):
        """Plot an image along with its histogram and cumulative histogram.

        """
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    def contrast_stretching(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        return img_rescale

    def equalization(self, image):
        img_eq = exposure.equalize_hist(image)
        return img_eq

    def adaptive_equalization(self, image):
        img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
        return img_adapteq

    def rescale_image(self, image):
        v_min, v_max = np.percentile(image, (0.2, 99.8))
        better_contrast = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        return better_contrast

    def otsu_threshold(self, image):
        val = filters.threshold_otsu(image)
        regions = np.digitize(image, bins=val)

    def histogram_run(self, image):
        image_rescale = self.contrast_stretching(image)
        #img = self.log_adjust(image_rescale)
        image_adapeq = self.equalization(image)
        #self.display_result(image, image_rescale, image_eq, image_adapeq)
        return image_adapeq

    def display_result(self, image, img_rescale, img_eq, img_adapteq):
        fig = plt.figure(figsize=(8, 5))
        axes = np.zeros((2, 4), dtype=np.object)
        axes[0, 0] = fig.add_subplot(2, 4, 1)
        for i in range(1, 4):
            axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
        for i in range(0, 4):
            axes[1, i] = fig.add_subplot(2, 4, 5 + i)

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(image, axes[:, 0])
        ax_img.set_title('Low contrast image')

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(img_rescale, axes[:, 1])
        ax_img.set_title('Contrast stretching')

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(img_eq, axes[:, 2])
        ax_img.set_title('Histogram equalization')

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(img_adapteq, axes[:, 3])
        ax_img.set_title('Adaptive equalization')

        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(np.linspace(0, 1, 5))

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()
