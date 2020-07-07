import configparser
import sys
from utils import str2bool
from preprocess import sequence2vector
import nngp


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    if str2bool(config['Pipeline']['vector']):
        print("Start to convert amino acid sequence to the vector")
        sequence2vector(config['Vector'], config['Files'])

    test_sets = ['Mamu-A1_001_01_9mer', 'HLA-A_02_01_9mer', 'B2705',
                'A0206_9mer', 'A6802_9mer']
    test_sets = ['data/' + i for i in test_sets]
    for test_set in test_sets:
        config['Files']['test_set'] = test_set
        if str2bool(config['Pipeline']['train']):
            print("Begin to train {}".format(test_set))
            nngp.train(config['Hyper-parameters for nngp'], config['Files'])
