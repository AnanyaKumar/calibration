
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')
parser.add_argument('--calibration_data_size', default=1000, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--bin_data_size', default=2000, type=int,
                    help='Number of examples to use for binning.')
# parser.add_argument('--num_bins', default=10, type=int,
# 					help='Bins to test estimators with.')





if __name__ == "__main__":
	args = parser.parse_args()
	logits, labels = utils.load_test_logits_labels(args.logits_file)
	lower_bound_experiment(logits, labels, args.platt_data_size, args.bin_data_size)
