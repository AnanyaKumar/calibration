
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


def noisy_platt_function(a, b, eps, l, u):
	def platt(x):
		x = np.log(x / (1 - x))
		x = a * x + b
		return 1 / (1 + np.exp(-x))
	assert(1 - eps >= platt(l) >= eps)
	assert(1 - eps >= platt(u) >= eps)
	bins = 100000
	noise = (np.random.binomial(1, np.ones(bins + 1) * 0.5) * 2 - 1) * eps
	def eval(x):
		assert l <= x <= u
		b = np.floor((x - l) / (u - l) * bins).astype(np.int32)
		assert(np.all(b <= bins))
		b -= (b == bins)
		return platt(x) + noise[b]
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


def sweep_n_platt(a, b, save_file):
	f = platt_function(a, b)
	Calibrators = [calibrators.HistogramCalibrator, calibrators.PlattBinnerCalibrator]
	dist = np.random.uniform
	nb_args = [(500 * i, 10) for i in range(1, 9)]
	num_trials = 1000
	num_evaluation = 10000
	means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
	pickle.dump((nb_args, means, stddevs), open(save_file, "wb"))


def sweep_b_platt(a, b, save_file):
	f = platt_function(a, b)
	Calibrators = [calibrators.HistogramCalibrator, calibrators.PlattBinnerCalibrator]
	dist = np.random.uniform
	nb_args = [(2000, 5 * i) for i in range(1, 9)]
	num_trials = 1000
	num_evaluation = 10000
	means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
	pickle.dump((nb_args, means, stddevs), open(save_file, "wb"))


def sweep_n_noisy_platt(a, b, save_file):
	l, u = 0.25, 0.75
	f = noisy_platt_function(a, b, eps=0.02, l=l, u=u)
	Calibrators = [calibrators.HistogramCalibrator, calibrators.PlattBinnerCalibrator, calibrators.PlattCalibrator]
	def dist(size):
		return np.random.uniform(low=l, high=u, size=size)
	nb_args = [(500 * i, 10) for i in range(1, 9)]
	num_trials = 1000
	num_evaluation = 10000
	means, stddevs = get_errors(f, Calibrators, dist, nb_args, num_trials, num_evaluation)
	print(means)
	print(stddevs)
	pickle.dump((nb_args, means, stddevs), open(save_file, "wb"))


def plot_sweep_n(load_file, hist_save_file, platt_save_file):
	(nb_args, means, stddevs) = pickle.load(open(load_file, "rb"))
	error_bars_90 = (1.645 / (np.square(means))) * stddevs
	plt.clf()
	plt.errorbar([n for (n, b) in nb_args], 1 / means[0], color='red',
			 yerr=[error_bars_90[0], error_bars_90[0]],
			 barsabove=True, capsize=4)
	plt.ylabel("1 / epsilon^2")
	plt.xlabel("n (number of calibration points)")
	plt.savefig(hist_save_file)
	plt.clf()
	plt.errorbar([n for (n, b) in nb_args], 1 / means[1], color='blue',
		     yerr=[error_bars_90[1], error_bars_90[1]],
		     barsabove=True, capsize=4)
	plt.ylabel("1 / epsilon^2")
	plt.xlabel("n (number of calibration points)")
	plt.tight_layout()
	plt.savefig(platt_save_file)
	print(means)
	print(stddevs)


def plot_sweep_b(load_file, hist_save_file, platt_save_file):
	(nb_args, means, stddevs) = pickle.load(open(load_file, "rb"))
	error_bars_90 = (1.645 / (np.square(means))) * stddevs
	plt.clf()
	plt.errorbar([b for (n, b) in nb_args], 1 / means[0], color='red',
			 yerr=[error_bars_90[0], error_bars_90[0]],
			 barsabove=True, capsize=4)
	plt.ylabel("1 / epsilon^2")
	plt.xlabel("b (number of bins)")
	plt.savefig(hist_save_file)
	plt.clf()
	plt.errorbar([b for (n, b) in nb_args], 1 / means[1], color='blue',
		     yerr=[error_bars_90[1], error_bars_90[1]],
		     barsabove=True, capsize=4)
	plt.ylabel("1 / epsilon^2")
	plt.xlabel("b (number of bins)")
	plt.tight_layout()
	plt.savefig(platt_save_file)
	print(means)
	print(stddevs)


def divide(mu1, mu2, sigma1, sigma2):
	mu = mu1 / mu2
	sigma = np.sqrt(1.0 / (mu2 ** 2) * (sigma1 ** 2) + (mu1 ** 2) / (mu2 ** 4) * (sigma2 ** 2))
	return mu, sigma


def plot_curve(f, save_file, l=1e-8, u=1.0-1e-8):
	xs = np.arange(l, u, 1 / 1000.0)
	ys = f(xs)
	plt.clf()
	plt.plot(xs, ys)
	plt.ylabel("P(Y = 1 | z)")
	plt.xlabel("z")
	plt.tight_layout()
	plt.savefig(save_file)


if __name__ == "__main__":
	# f = platt_function(1, 0)
	# plot_curve(f, './saved_files/curve_vary_n_a1_b0')
	# sweep_n_platt(1, 0, './saved_files/vary_n_a1_b0')
	# plot_sweep_n(load_file='./saved_files/vary_n_a1_b0',
	# 	         hist_save_file='./saved_files/hist_vary_n_a1_b0',
	# 	         platt_save_file='./saved_files/platt_vary_n_a1_b0')

	# f = platt_function(1, 0)
	# plot_curve(f, './saved_files/curve_vary_b_a1_b0')
	# sweep_b_platt(1, 0, './saved_files/vary_b_a1_b0')
	# plot_sweep_b(load_file='./saved_files/vary_b_a1_b0',
	# 	         hist_save_file='./saved_files/hist_vary_b_a1_b0',
	# 	         platt_save_file='./saved_files/platt_vary_b_a1_b0')

	# f = platt_function(2, 1)
	# plot_curve(f, './saved_files/curve_vary_n_a2_b1')
	# sweep_n_platt(2, 1, './saved_files/vary_n_a2_b1')
	# plot_sweep_n(load_file='./saved_files/vary_n_a2_b1',
	# 	         hist_save_file='./saved_files/hist_vary_n_a2_b1',
	# 	         platt_save_file='./saved_files/platt_vary_n_a2_b1')

	f = platt_function(2, 1)
	plot_curve(f, './saved_files/curve_vary_b_a2_b1')
	# sweep_b_platt(2, 1, './saved_files/vary_b_a2_b1')
	# plot_sweep_b(load_file='./saved_files/vary_b_a2_b1',
	# 	         hist_save_file='./saved_files/hist_vary_b_a2_b1',
	# 	         platt_save_file='./saved_files/platt_vary_b_a2_b1')

	f = noisy_platt_function(2, 1, eps=0.02, l=0.25, u=0.75)
	plot_curve(f, './saved_files/noise_curve_vary_n_a2_b1', l=0.25, u=0.75)
	sweep_n_noisy_platt(2, 1, './saved_files/noise_vary_n_a2_b1')
