import os
import glob
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utility.histogram_processor import HistogramShow
from Utility.image_loader import ImageLoader
from Utility.gabor import GaborExtractFeatures
from Utility.csv_to_arff import ArffFileSaver


class Analyzer(HistogramShow, ImageLoader, GaborExtractFeatures):
    SEGMENTED_RIGHT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Right/*.bmp"
    SEGMENTED_LEFT_HANDS = "Hands Database/IITD Palmprint V1/Segmented/Left/*.bmp"
    SAVE_LOCATION = "Histogram_processed/{}.bmp"
    PROCESSED_IMAGES_PATH = "Histogram_processed/*.bmp"

    def gabor_filter(self, both_hands, left_hand, right_hand, train_test_ratio):
        """
        Method that is starting the image enhancement and applies gabor filter
        :param both_hands: <bool> if True, both left and right hand images will be processed
        :param left_hand: <bool> if True, only left images will be processed
        :param right_hand: <bool> if True, only right hand images will be processed
        :param train_test_ratio: <str> ration between train images and test images e.g.: 3/2 is 3 train images, 2 test
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
        with open('Gabor_train.csv', 'w') as fp:
            fp.truncate()
        with open('Gabor_test.csv', 'w') as fp:
            fp.truncate()
        # self.display_result2('001_1_seg.bmp')
        # self.display_result2('001_1_pro.bmp')

        if both_hands is True:
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
        elif left_hand is True or right_hand is True:
            counter = []
            for index, image in enumerate(self.imgs):
                img_to_be_saved = self.histogram_run(image)
                self.base[index] = self.base[index].replace(".bmp", "")
                counter.append(self.base[index].rsplit('\\', 1)[-1])
                self.save_image(img_to_be_saved, self.SAVE_LOCATION.format(counter[index]))

        print ("Applying gabor filter on processed images!")
        self.gabor_plot(processed_images_path=self.PROCESSED_IMAGES_PATH, train_test_ratio=train_test_ratio,
                        both_images=both_hands)

    def main(self, args):
        """ Setting arguments for gabor method """
        parser = argparse.ArgumentParser(description="Type all parameters")
        parser.add_argument("both_hands", help="both_hands", type=self.str2bool, default=False)
        parser.add_argument("--left_hand", help="left_hand", type=self.str2bool, default=False, required=False)
        parser.add_argument("--right_hand", help="right_hand", type=self.str2bool, default=False, required=False)
        parser.add_argument("--train_test_ratio", help="train_test_ratio", type=str, default="3/2", required=False)
        args = parser.parse_args(args)
        self.gabor_filter(args.both_hands, args.left_hand, args.right_hand, args.train_test_ratio)
        ArffFileSaver(args.both_hands, args.left_hand, args.right_hand, args.train_test_ratio)


if __name__ == "__main__":
    Analyzer().main(sys.argv[1:])
    # ArffFileSaver()
    print ("All images were processed!")
