import numpy as np

from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader
from skimage import img_as_ubyte, img_as_float, img_as_uint

class Analyzer(HistogramShow, ImageLoader):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/001_1.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"
    SAVE = "Histogram_processed/File_{}.bmp"

    def run_now(self):
        # Load an example image
        img_collection = self.add_images_to_collection(self.SEGMENTED_RIGHT_HANDS)

        for index, image in enumerate(img_collection):

            print img_as_float(image)

            self.IMG = self.run(image)

            print self.IMG

            self.save_image(img_as_uint(self.IMG), self.SAVE.format(index))

if __name__== "__main__":
    test  = Analyzer()
    test.run_now()
