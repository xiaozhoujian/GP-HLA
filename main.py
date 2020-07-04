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

    if str2bool(config['Pipeline']['train']):
        print("Begin to train")
        nngp.train(config['Hyper-parameters for nngp'], config['Files'])
