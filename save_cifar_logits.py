
import argparse
from tensorflow.keras.datasets import cifar10

import cifar10vgg
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--save_file', default='cifar_logits.dat', type=str,
                    help='Name of file to save logits, labels pair.')

if __name__ == "__main__":
	args = parser.parse_args()
	utils.save_test_logits_labels(cifar10, cifar10vgg.cifar10vgg(), args.save_file)
