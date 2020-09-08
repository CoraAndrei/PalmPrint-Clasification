import csv
from time import sleep
import os


class ArffFileSaver(object):

    def __init__(self, both_hands=False, left_hand=False, right_hand=False, train_test_ratio='4/1',
                 file=['Gabor_test.csv', 'Gabor_train.csv']):
        for index in file:
            self.csvInput(index,  both_hands, left_hand, right_hand, train_test_ratio)
            print 'Finished.'

    # import CSV
    def csvInput(self, file,  both_hands=False, left_hand=False, right_hand=False, train_test_ratio='4/1'):
        self.content = []
        self.name = ''
        user = file

        d = os.path.dirname(__file__)  # directory of script
        path = r'{}/Gabor_files/'.format(d)  # path to be created
        try:
            os.makedirs(path)
        except OSError:
            pass

        # remove .csv
        if user.endswith('.csv') == True:
            self.name = user.replace('.csv', '')

        with open(user, 'rb') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                self.content.append(row)
        csvfile.close()
        sleep(2)  # sleeps added for dramatic effect!

    # export ARFF
        print 'Save to ARFF file.'
        if both_hands:
            self.end_title = 'bimodal'
        elif left_hand:
            self.end_title = 'unimodal_left_hand'
        elif right_hand:
            self.end_title = 'unimodal_right_hand'
        if train_test_ratio == '4/1':
            self.ratio = '4.1'
        elif train_test_ratio == '3/2':
            self.ratio = '3.2'
        title = path + str(self.name) + '_' + self.end_title + '_' + self.ratio +'.arff'
        new_file = open(title, 'w')

        ##
        # following portions formats and writes to the new ARFF file
        ##

        # write relation
        new_file.write('@relation ' + str(self.name) + '\n\n')

        # get attribute type input
        for i in range(len(self.content[0]) - 1):
            attribute_type = 'numeric'
            new_file.write('@attribute ' + str(self.content[0][i]) + ' ' + str(attribute_type) + '\n')

        # create list for class attribute
        last = len(self.content[0])
        class_items = []
        for i in range(len(self.content)):
            name = self.content[i][last - 1]
            if name not in class_items:
                class_items.append(self.content[i][last - 1])
            else:
                pass
        del class_items[0]

        string = '{' + ','.join(sorted(class_items)) + '}'
        new_file.write('@attribute ' + str(self.content[0][last - 1]) + ' ' + str(string) + '\n')

        # write data
        new_file.write('\n@data\n')

        del self.content[0]
        for row in self.content:
            new_file.write(','.join(row) + '\n')

        # close file
        new_file.close()
        sleep(2)

