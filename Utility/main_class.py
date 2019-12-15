
from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader

class Analyzer(HistogramShow, ImageLoader):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/*.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"

    def run_now(self):
        # Load an example image
        img_collection = self.add_images_to_collection(self.SEGMENTED_RIGHT_HANDS)

        for image in img_collection:
            self.run(image)

if __name__== "__main__":
    test  = Analyzer()
    test.run_now()
