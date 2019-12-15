from Utility.histogram_processor import HistogramShow as hist
from Utility.image_loader import ImageLoader as img_load

class HistogramBulkProcessor(object):

    def bulk_hist_show(self, path):
        my_images_collection = img_load.add_images_to_collection(path)

        for index in my_images_collection:
            hist.plot_img_and_hist(index)

