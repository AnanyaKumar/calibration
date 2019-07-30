
import numpy as np
import utils
import calibrators
import matplotlib.pyplot as plt
import pickle

def platt_function(a, b):
	def eval(x):
		x = np.log(x / (1 - x))
		x = a * x + b
		return 1 / (1 + np.exp(-x))
	return np.vectorize(eval)

def sample(function, dist, n):
	zs = dist(size=n)
	ps = function(zs)
	ys = np.random.binomial(1, p=ps)
	return (zs, ys)

def evaluate(function, calibrator, dist, n):
	zs = dist(size=n)
	ps = function(zs)
	phats = calibrator.calibrate(zs)
	bins = utils.get_discrete_bins(phats)
	data = list(zip(phats, ps))
	binned_data = utils.bin(data, bins)
	return utils.plugin_ce(binned_data) ** 2

def evaluate_mse(function, calibrator, dist, n):
	zs = dist(size=n)
	ps = function(zs)
	phats = calibrator.calibrate(zs)
	return np.mean(np.square(ps - phats))

# Given calibrators, f, and a list of n, B arguments, num_trials, produce a list of calbration errors and std-devs
def get_errors(function, Calibrators, dist, nb_args, num_trials, num_evaluation):
	means = np.zeros((len(Calibrators), len(nb_args)))
	std_devs = np.zeros((len(Calibrators), len(nb_args)))
	for i, Calibrator in zip(range(len(Calibrators)), Calibrators):
		for j, (num_calibration, num_bins) in zip(range(len(nb_args)), nb_args):
			current_errors = []
			for k in range(num_trials):
				zs, ys = sample(function, dist=dist, n=num_calibration)
				calibrator = Calibrator(num_calibration=num_calibration, num_bins=num_bins)
				calibrator.train_calibration(zs, ys)
				error = evaluate(function, calibrator, dist, n=num_evaluation)
				assert(error >= 0.0)
				current_errors.append(error)
			means[i][j] = np.mean(current_errors)
			std_devs[i][j] = np.std(current_errors) / np.sqrt(num_trials)
	return means, std_devs


def generate_data(a, b, save_file):
	f = platt_function(1, 0)
	Calibrators = [calibrators.HistogramCalibrator, calibrators.PlattBinnerCalibrator]
	dist = np.random.uniform
	nb_args = [(1000, 10), (2000, 10), (3000, 10), (4000, 10), (5000, 10), (6000, 10)]
	# nb_args = [(500, 10), (500, 20), (500, 30), (500, 40), (500, 50), (500, 60), (500, 70), (500, 80)]
	num_trials = 100
	num_evaluation = 10000
	save_file = './saved_files/vary_n_a1_b1'
	means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
	pickle.dump((nb_args, means, stddevs), open(save_file, "wb"))

	# plt.plot([n for (n, b) in nb_args], 1 / means[0])
	# plt.show()
	# plt.plot([n for (n, b) in nb_args], 1 / means[1])
	# plt.show()
	# print(means)

	# f = platt_function(1, 0)
	# Calibrators = [calibrators.HistogramCalibrator, calibrators.PlattBinnerCalibrator]
	# dist = np.random.uniform
	# nb_args = [(500, 10), (500, 20), (500, 30), (500, 40), (500, 50), (500, 60), (500, 70), (500, 80)]
	# num_trials = 10
	# num_evaluation = 10000
	# save_file = './saved_files/vary_b_a1_b1'
	# means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
	# print(means)
	# print(stddevs)
	

	# zs, ys = sample(f, dist=np.random.uniform, n=num_calibration)
	# hist = calibrators.HistogramCalibrator(num_calibration=num_calibration, num_bins=num_bins)
	# platt_bin = calibrators.PlattBinnerCalibrator(num_calibration=num_calibration, num_bins=num_bins)
	# hist.train_calibration(zs, ys)
	# platt_bin.train_calibration(zs, ys)
	# print(evaluate(f, hist, dist=np.random.uniform, n=num_evaluation))
	# print(evaluate(f, platt_bin, dist=np.random.uniform, n=num_evaluation))
	# Sample N points from the platt function
	# Fit calibrators
	# Evaluate the calibrators

def plot():
	save_file = './saved_files/vary_n_a1_b1'
	(nb_args, means, stddevs) = pickle.load(open(save_file, "rb"))
	plt.plot([n for (n, b) in nb_args], 1 / means[0])
	plt.show()
	plt.plot([n for (n, b) in nb_args], 1 / means[1])
	plt.show()
	print(means)

if __name__ == "__main__":
	generate_data()
	plot()
