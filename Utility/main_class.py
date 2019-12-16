from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader

class Analyzer(HistogramShow, ImageLoader):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/*.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"
    SAVE = "Histogram_processed/File_{}.bmp"

    def run_now(self):
        # Load images
        img_collection = self.add_images_to_collection(self.SEGMENTED_RIGHT_HANDS)

        for index, image in enumerate(img_collection):
            img_to_be_saved = self.run(image)
            self.save_image((img_to_be_saved), self.SAVE.format(index))


if __name__== "__mai0n__":
     Analyzer().run_now()
