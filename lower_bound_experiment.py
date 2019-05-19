
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')
parser.add_argument('--calibration_data_size', default=1000, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--bin_data_size', default=1000, type=int,
                    help='Number of examples to use for binning.')
parser.add_argument('--plot_save_file', default='lower_bound_plot.png', type=str,
                    help='File to save lower bound plot.')


def lower_bound_experiment(logits, labels, calibration_data_size, bin_data_size, bins_list,
					       save_name='cmp_est', binning_func=utils.get_equal_bins, lp=2):
	# Shuffle the logits and labels.
	indices = np.random.choice(list(range(len(logits))), size=len(logits), replace=False)
	logits = [logits[i] for i in indices]
	labels = [labels[i] for i in indices]
	predictions = utils.get_top_predictions(logits)
	probs = utils.get_top_probs(logits)
	correct = (predictions == labels)
	print('num_correct: ', sum(correct))
	# Platt scale on first chunk of data
	platt = utils.get_platt_scaler(probs[:calibration_data_size], correct[:calibration_data_size])
	platt_probs = platt(probs)
	lower, middle, upper = [], [], []
	for num_bins in bins_list:
		bins = binning_func(
			platt_probs[:calibration_data_size+bin_data_size], num_bins=num_bins)
		verification_probs = platt_probs[calibration_data_size+bin_data_size:]
		verification_correct = correct[calibration_data_size+bin_data_size:]
		verification_data = list(zip(verification_probs, verification_correct))
		def estimator(data):
			binned_data = utils.bin(data, bins)
			return utils.plugin_ce(binned_data, power=lp)
		print('estimate: ', estimator(verification_data))
		estimate_interval = utils.bootstrap_uncertainty(
			verification_data, estimator, num_samples=1000)
		lower.append(estimate_interval[0])
		middle.append(estimate_interval[1])
		upper.append(estimate_interval[2])
		print('interval: ', estimate_interval)
	lower_errors = np.array(middle) - np.array(lower)
	upper_errors = np.array(upper) - np.array(middle)
	plt.clf()
	font = {'family' : 'normal', 'size': 18}
	rc('font', **font)
	plt.errorbar(
		bins_list, middle, yerr=[lower_errors, upper_errors],
		barsabove=True, fmt = 'none', color='black', capsize=4)
	plt.scatter(bins_list, middle, color='black')
	plt.xlabel(r"No. of bins")
	if lp == 2:
		plt.ylabel("Calibration error")
	else:
		plt.ylabel("L%d Calibration error" % lp)
	plt.xscale('log', basex=2)
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.tight_layout()
	plt.savefig(save_name)


def cifar_experiment(savefile, binning_func=utils.get_equal_bins, lp=2):
	np.random.seed(0)
	calibration_data_size = 1000
	bin_data_size = 1000
	logits, labels = utils.load_test_logits_labels('cifar_logits.dat')
	lower_bound_experiment(logits, labels, calibration_data_size, bin_data_size,
						   bins_list=[2, 4, 8, 16, 32, 64, 128], save_name=savefile,
						   binning_func=binning_func, lp=lp)


def imagenet_experiment(savefile, binning_func=utils.get_equal_bins, lp=2):
	np.random.seed(0)
	calibration_data_size = 20000
	bin_data_size = 5000
	logits, labels = utils.load_test_logits_labels('imagenet_logits.dat')
	lower_bound_experiment(logits, labels, calibration_data_size, bin_data_size,
						   bins_list=[2, 4, 8, 16, 32, 64, 128, 256, 512], save_name=savefile,
						   binning_func=binning_func, lp=lp)


if __name__ == "__main__":
	# cifar_experiment('l2_lower_bound_cifar_plot.png')
	cifar_experiment('l1_lower_bound_cifar_plot.png', lp=1)
	cifar_experiment('l1_lower_bound_cifar_plot_prob_bin.png',
				     binning_func=utils.get_equal_prob_bins, lp=1)
	# imagenet_experiment('l2_lower_bound_imagenet_plot.png')
	# imagenet_experiment('l1_lower_bound_imagenet_plot.png', lp=1)
	# imagenet_experiment('l1_lower_bound_imagenet_plot_prob_bin.png',
	# 					binning_func=utils.get_equal_prob_bins, lp=1)
	# args = parser.parse_args()
	# np.random.seed(0)
	# logits, labels = utils.load_test_logits_labels(args.logits_file)
	# lower_bound_experiment(logits, labels, args.calibration_data_size, args.bin_data_size,
	# 					   bins_list=[2, 4, 8, 16, 32, 64], save_name=args.plot_save_file)
