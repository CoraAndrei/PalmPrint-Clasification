import os
import glob
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader
from Utility.gabor import GaborExtractFeatures


class Analyzer(HistogramShow, ImageLoader, GaborExtractFeatures):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/*.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"
    SAVE_LOCATION = "Histogram_processed/{}.bmp"
    PROCESSED_IMAGES_PATH = "Histogram_processed/*.bmp"

    def gabor_filter(self, both_hands=True, left_hand=False, right_hand=False):
        """
        Method that is starting the image enhancement and applies gabor filter
        :param both_hands: <bool> if True, both left and right hand images will be processed
        :param left_hand: <bool> if True, only left images will be processed
        :param right_hand: <bool> if True, only right hand images will be processed
        :return:
        """
        print ("Deleting old processed images!")
        processed_images = glob.glob(self.PROCESSED_IMAGES_PATH)
        for f in processed_images:
            os.remove(f)

        right_imgs = self.add_images_to_collection(self.SEGMENTED_RIGHT_HANDS)
        right_imgs_base_path = glob.glob(self.SEGMENTED_RIGHT_HANDS)

        left_imgs = self.add_images_to_collection(self.SEGMENTED_LEFT_HANDS)
        left_imgs_base_path = glob.glob(self.SEGMENTED_LEFT_HANDS)

        if right_hand:
            self.imgs = right_imgs
            self.base = right_imgs_base_path
        elif left_hand:
            self.imgs = left_imgs
            self.base = left_imgs_base_path

        print ("Deleting old data from csv files!")
        with open('Gabor_learn.csv', 'w') as fp:
            fp.truncate()
        with open('Gabor_test.csv', 'w') as fp:
            fp.truncate()

        if both_hands:
            imgs = right_imgs
            base = right_imgs_base_path
            hands_counter = 2
            while hands_counter > 0:
                counter = []
                for index, image in enumerate(imgs):
                    img_to_be_saved = self.histogram_run(image)
                    base[index] = base[index].replace(".bmp", "")
                    counter.append(base[index].rsplit('\\', 1)[-1])
                    if hands_counter == 2:
                        self.save_image(img_to_be_saved, self.SAVE_LOCATION.format(counter[index]))
                    elif hands_counter == 1:
                        self.save_image(img_to_be_saved, self.SAVE_LOCATION.format(counter[index]+"a"))
                hands_counter -= 1
                imgs = left_imgs
                base = left_imgs_base_path
        elif left_hand or right_hand:
            counter = []
            for index, image in enumerate(self.imgs):
                img_to_be_saved = self.histogram_run(image)
                self.base[index] = self.base[index].replace(".bmp", "")
                counter.append(self.base[index].rsplit('\\', 1)[-1])
                self.save_image(img_to_be_saved, self.SAVE_LOCATION.format(counter[index]))

        print ("Applying gabor filter on processed images!")
        self.gabor_plot(processed_images_path=self.PROCESSED_IMAGES_PATH, both=both_hands)

    def main(self, args):
        parser = argparse.ArgumentParser(description="Type all parameters")
        parser.add_argument("both_hands", help="both_hands", type=str, default=True)
        parser.add_argument("--left_hand", help="left_hand", type=str, default=False, required=False)
        parser.add_argument("--right_hand", help="right_hand", type=str, default=False, required=False)

        args = parser.parse_args(args)
        self.gabor_filter(args.both_hands, args.left_hand, args.right_hand)


if __name__ == "__main__":
    Analyzer().main(sys.argv[1:])
    print ("All images were processed!")
