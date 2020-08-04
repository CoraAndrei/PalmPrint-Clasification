from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader
from Utility.gabor import GaborExtractFeatures
import glob


class Analyzer(HistogramShow, ImageLoader, GaborExtractFeatures):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/*.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"
    SEGMENTED_RIGHT_HAND_ONE = "Hands Database/IITD Palmprint V1/Segmented/Right/001_1.bmp"
    SAVE_LOCATION = "Histogram_processed/{}.bmp"

    PROCESSED_IMAGES_PATH = "Histogram_processed/*.bmp"

    def run_now(self):
        # Load images
        imgs = self.add_images_to_collection(self.SEGMENTED_RIGHT_HANDS)
        base = glob.glob("Hands Database\IITD Palmprint V1\Segmented\Right\*.bmp")

        with open('Gabor_results.csv', 'w') as fp:
            fp.truncate()

        counter = []
        for index, image in enumerate(imgs):
            img_to_be_saved = self.histogram_run(image)
            base[index] = base[index].replace(".bmp", "")
            counter.append(base[index].rsplit('\\', 1)[-1])
            self.save_image(img_to_be_saved, self.SAVE_LOCATION.format(counter[index]))

        self.gabor_plot(processed_images_path=self.PROCESSED_IMAGES_PATH)


if __name__== "__main__":
    Analyzer().run_now()
