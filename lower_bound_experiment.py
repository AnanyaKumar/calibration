
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
# parser.add_argument('--num_bins', default=10, type=int,
# 					help='Bins to test estimators with.')


def lower_bound_experiment(logits, labels, calibration_data_size, bin_data_size, bins_list,
					       save_name='cmp_est'):
	# Calibrate using Platt/temperature scaling
	predictions = utils.get_top_predictions(logits)
	probs = utils.get_top_probs(logits)
	correct = (predictions == labels)
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
			return utils.plugin_ce(binned_data, power=1)
		print('estimate: ', estimator(verification_data))
		estimate_interval = utils.bootstrap_uncertainty(
			verification_data, estimator, num_samples=100)
		lower.append(estimate_interval[0])
		middle.append(estimate_interval[1])
		upper.append(estimate_interval[2])
		print('interval: ', estimate_interval)
	lower_errors = np.array(middle) - np.array(lower)
	upper_errors = np.array(upper) - np.array(middle)
	x_axis = np.log(bins_list)/np.log(2)
	rc('font',**{'size': 12, 'family':'sans-serif','sans-serif':['Arial']})
	rc('text', usetex=True)
	plt.errorbar(
		x_axis, middle, yerr=[lower_errors, upper_errors],
		barsabove=True, fmt = 'none', color='black', capsize=4)
	plt.scatter(x_axis, middle, color='black')
	plt.xlabel(r"$\log_2(\mbox{no. of bins})$")
	plt.ylabel(r"$\ell_1-\mbox{calibration error}$")
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.set_xlim([x_axis[0] - 0.99,x_axis[-1]+0.5])
	plt.show()


if __name__ == "__main__":
	args = parser.parse_args()
	logits, labels = utils.load_test_logits_labels(args.logits_file)
	lower_bound_experiment(logits, labels, args.calibration_data_size, args.bin_data_size,
						   bins_list=[2, 4, 8, 16, 32, 64])
