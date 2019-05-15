
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
# parser.add_argument('--num_bins', default=10, type=int,
# 					help='Bins to test estimators with.')


def lower_bound_experiment(logits, labels, calibration_data_size, bin_data_size, bins_list,
					       save_name='cmp_est'):
	# Calibrate using Platt/temperature scaling
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
		bins = utils.get_equal_bins(
			platt_probs[:calibration_data_size+bin_data_size], num_bins=num_bins)
		verification_probs = platt_probs[calibration_data_size+bin_data_size:]
		verification_correct = correct[calibration_data_size+bin_data_size:]
		verification_data = list(zip(verification_probs, verification_correct))
		def estimator(data):
			binned_data = utils.bin(data, bins)
			return utils.plugin_ce(binned_data, power=2)
		print('estimate: ', estimator(verification_data))
		estimate_interval = utils.bootstrap_uncertainty(
			verification_data, estimator, num_samples=100)
		lower.append(estimate_interval[0])
		middle.append(estimate_interval[1])
		upper.append(estimate_interval[2])
		print('interval: ', estimate_interval)
	lower_errors = np.array(middle) - np.array(lower)
	upper_errors = np.array(upper) - np.array(middle)
	# x_axis = np.log(bins_list)/np.log(2)
	font = {'family' : 'normal', 'size': 18}
	rc('font', **font)
	plt.errorbar(
		bins_list, middle, yerr=[lower_errors, upper_errors],
		barsabove=True, fmt = 'none', color='black', capsize=4)
	plt.scatter(bins_list, middle, color='black')
	plt.xlabel(r"No. of bins")
	plt.ylabel(r"Calibration error")
	plt.xscale('log', basex=2)
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	# ax.set_xlim([x_axis[0] - 0.99,x_axis[-1]+0.5])
	plt.tight_layout()
	plt.savefig(args.plot_save_file)


if __name__ == "__main__":
	args = parser.parse_args()
	logits, labels = utils.load_test_logits_labels(args.logits_file)
	lower_bound_experiment(logits, labels, args.calibration_data_size, args.bin_data_size,
						   bins_list=[2, 4, 8, 16, 32, 64])
