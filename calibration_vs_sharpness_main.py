
import argparse
import calibrators
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import time


import utils

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')
parser.add_argument('--num_bin_selection', default=500, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--num_binning', default=500, type=int,
                    help='Number of examples to use for binning.')


def eval_top_calibration(probs, logits, labels):
    correct = (utils.get_top_predictions(logits) == labels)
    data = list(zip(probs, correct))
    bins = utils.get_discrete_bins(probs)
    binned_data = utils.bin(data, bins)
    return utils.plugin_ce(binned_data) ** 2


def eval_marginal_calibration(probs, logits, labels, plugin=True):
    ces = []  # Compute the calibration error per class, then take the average.
    k = logits.shape[1]
    labels_one_hot = utils.get_labels_one_hot(np.array(labels), k)
    for c in range(k):
        probs_c = probs[:, c]
        labels_c = labels_one_hot[:, c]
        data_c = list(zip(probs_c, labels_c))
        bins_c = utils.get_discrete_bins(probs_c)
        binned_data_c = utils.bin(data_c, bins_c)
        if plugin:
            ce_c = utils.plugin_ce(binned_data_c) ** 2
        else:
            ce_c = utils.unbiased_square_ce(binned_data_c)
        ces.append(ce_c)
    return np.mean(ces)


def upper_bound_marginal_calibration_unbiased(probs, logits, labels, samples=30):
    data = list(zip(probs, logits, labels))
    def evaluator(data):
        probs, logits, labels = list(zip(*data))
        probs, logits, labels = np.array(probs), np.array(logits), np.array(labels)
        return eval_marginal_calibration(probs, logits, labels, plugin=False)
    estimate = evaluator(data)
    conf_interval = utils.bootstrap_std(data, evaluator, num_samples=samples)
    return estimate + 1.3 * conf_interval


def upper_bound_marginal_calibration_biased(probs, logits, labels, samples=30):
    data = list(zip(probs, logits, labels))
    def evaluator(data):
        probs, logits, labels = list(zip(*data))
        probs, logits, labels = np.array(probs), np.array(logits), np.array(labels)
        return eval_marginal_calibration(probs, logits, labels, plugin=True)
    estimate = evaluator(data)
    conf_interval = utils.bootstrap_std(data, evaluator, num_samples=samples)
    return estimate + 1.3 * conf_interval


def compare_calibrators(data_sampler, num_bins, Calibrators, calibration_evaluators,
                        eval_mse):
    """Get one sample of the calibration error and MSE for a set of calibrators.

    Args:
        data_sampler: A function that takes in 0 arguments
            and returns calib_logits, calib_labels, eval_logits, eval_labels, mse_logits,
            mse_labels, where calib_logits and calib_labels should be used by the calibrator
            to calibrate, eval_logits and eval_labels should be used to measure the calibration
            error, and mse_logits, mse_labels should be used to measure the mean-squared error.
        num_bins: integer number of bins.
        Calibrators: calibrator classes from e.g. calibrators.py.
        calibration_evaluators: a list of functions. calibration_evaluators[i] takes
            the output from the calibration method of calibrator i, eval_logits,
            eval_labels, and returns a float representing the calibration error
            (or an upper bound of it) of calibrator i. We suppose multiple calibration
            evaluators because different calibrators may require different ways
            of estimating/upper bounding calibration error.
        eval_mse: a function that takes in the output of the calibration method,
            mse_logits, mse_labels, and returns a float representing the MSE.
    """
    calib_logits, calib_labels, eval_logits, eval_labels, mse_logits, mse_labels = data_sampler()
    l2_ces = []
    mses = []
    train_time = 0.0
    eval_time = 0.0
    start_total = time.time()
    for Calibrator, i in zip(Calibrators, range(len(Calibrators))):
        calibrator = Calibrator(1, num_bins)
        start_time = time.time()
        calibrator.train_calibration(calib_logits, calib_labels)
        train_time += (time.time() - start_time)
        calibrated_probs = calibrator.calibrate(eval_logits)
        start_time = time.time()
        mid = calibration_evaluators[i](calibrated_probs, eval_logits, eval_labels)
        eval_time += time.time() - start_time
        cal_mse_logits = calibrator.calibrate(mse_logits)
        mse = eval_mse(cal_mse_logits, mse_logits, mse_labels)
        l2_ces.append(mid)
        mses.append(mse)
    print('train_time: ', train_time)
    print('eval_time: ', eval_time)
    print('total_time: ', time.time() - start_total)
    return l2_ces, mses


def average_calibration(data_sampler, num_bins, Calibrators, calibration_evaluators,
                        eval_mse, num_trials=100):
    l2_ces, mses = [], []
    for i in range(num_trials):
        print(i)
        cur_l2_ces, cur_mses = compare_calibrators(
            data_sampler, num_bins, Calibrators,
            calibration_evaluators, eval_mse)
        l2_ces.append(cur_l2_ces)
        mses.append(cur_mses)
    l2_ce_means = np.mean(l2_ces, axis=0)
    l2_ce_stddevs = np.std(l2_ces, axis=0) / np.sqrt(num_trials)
    mses = np.mean(mses, axis=0)
    mse_stddevs = np.std(mses, axis=0) / np.sqrt(num_trials)
    return l2_ce_means, l2_ce_stddevs, mses, mse_stddevs


def vary_bin_calibration(data_sampler, num_bins_list, Calibrators, calibration_evaluators,
                         eval_mse, num_trials=100):
    ce_list = []
    stddev_list = []
    mse_list = []
    for num_bins in num_bins_list:
        l2_ce_means, l2_ce_stddevs, mses, mse_stddevs = average_calibration(
            data_sampler, num_bins, Calibrators,
            calibration_evaluators, eval_mse, num_trials)
        ce_list.append(l2_ce_means)
        stddev_list.append(l2_ce_stddevs)
        mse_list.append(mses)
    return np.transpose(ce_list), np.transpose(stddev_list), np.transpose(mse_list)


def plot_ces(bins_list, l2_ces, l2_ce_stddevs):
    font = {'family' : 'normal',
        'size'   : 20}
    rc('font', **font)
    # 90% confidence intervals.
    error_bars_90 = 1.645 * l2_ce_stddevs
    plt.errorbar(
      bins_list, l2_ces[0], yerr=[error_bars_90[0], error_bars_90[0]],
      barsabove=True, color='red', capsize=4, label='histogram')
    plt.errorbar(
      bins_list, l2_ces[1], yerr=[error_bars_90[1], error_bars_90[1]],
      barsabove=True, color='blue', capsize=4, label='variance-reduced')
    plt.ylabel("L2 Squared Calibration Error")
    plt.xlabel("Number of Bins")
    plt.legend(loc='lower right')
    plt.savefig('marginal_ces.png')


def plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=None, ylim=None):
    font = {'family' : 'normal',
        'size'   : 20}
    rc('font', **font)
    def get_pareto_points(data):
        pareto_points = []
        def dominated(p1, p2):
            return p1[0] >= p2[0] and p1[1] >= p2[1]
        for datum in data:
            num_dominated = sum(map(lambda x: dominated(datum, x), data))
            if num_dominated == 1:
                pareto_points.append(datum)
        return pareto_points
    print(get_pareto_points(list(zip(l2_ces[0], mses[0], bins_list))))
    print(get_pareto_points(list(zip(l2_ces[1], mses[1], bins_list))))
    l2ces0, mses0 = zip(*get_pareto_points(list(zip(l2_ces[0], mses[0]))))
    l2ces1, mses1 = zip(*get_pareto_points(list(zip(l2_ces[1], mses[1]))))
    plt.title("MSE vs Calibration Error")
    plt.scatter(l2ces0, mses0, c='red', marker='o', label='hist')
    plt.scatter(l2ces1, mses1, c='blue', marker='s', label='ours')
    plt.legend(loc='upper left')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("L2 Squared Calibration Error")
    plt.ylabel("Mean-Squared Error")
    plt.savefig('marginal_mse_vs_ces.png')


def make_calibration_data_sampler(logits, labels, num_calibration):
    def data_sampler():
        assert len(logits) == len(labels)
        indices = np.random.choice(list(range(len(logits))),
                                   size=num_calibration, replace=True)
        calib_logits = np.array([logits[i] for i in indices])
        calib_labels = np.array([labels[i] for i in indices])
        eval_logits = logits
        eval_labels = labels
        return calib_logits, calib_labels, eval_logits, eval_labels, logits, labels
    return data_sampler


def make_calibration_eval_data_sampler(logits, labels, num_calib, num_eval):
    def data_sampler():
        assert len(logits) == len(labels)
        calib_indices = np.random.choice(
            list(range(len(logits))), size=num_calib, replace=True)
        eval_indices = np.random.choice(
            list(range(len(logits))), size=num_eval, replace=True)
        calib_logits = np.array([logits[i] for i in calib_indices])
        calib_labels = np.array([labels[i] for i in calib_indices])
        eval_logits = np.array([logits[i] for i in eval_indices])
        eval_labels = np.array([labels[i] for i in eval_indices])
        return calib_logits, calib_labels, eval_logits, eval_labels, logits, labels
    return data_sampler


def cifar10_experiment_top_1_1_1000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramTopCalibrator, calibrators.PlattBinnerTopCalibrator],
        calibration_evaluators=[eval_top_calibration, eval_top_calibration],
        eval_mse=utils.eval_top_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.002), ylim=(0.0425, 0.045))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_top_1_2_3000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 3000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramTopCalibrator, calibrators.PlattBinnerTopCalibrator],
        calibration_evaluators=[eval_top_calibration, eval_top_calibration],
        eval_mse=utils.eval_top_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.002), ylim=(0.0425, 0.045))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_marginal_2_1_1000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 20
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        calibration_evaluators=[eval_marginal_calibration, eval_marginal_calibration],
        eval_mse=utils.eval_marginal_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.001), ylim=(0.0, 0.07))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_marginal_2_2_3000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 20
    num_calibration = 3000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        calibration_evaluators=[eval_marginal_calibration, eval_marginal_calibration],
        eval_mse=utils.eval_marginal_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.001), ylim=(0.0, 0.04))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_marginal_3_1_1000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calib = 1000
    num_eval = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_eval_data_sampler(logits, labels, num_calib, num_eval),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        calibration_evaluators=[upper_bound_marginal_calibration_biased,
                                upper_bound_marginal_calibration_unbiased],
        eval_mse=utils.eval_marginal_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.002), ylim=(0.0, 0.04))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def imagenet_experiment_top_1_1_1000():
    logits_file = 'imagenet_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramTopCalibrator, calibrators.PlattBinnerTopCalibrator],
        calibration_evaluators=[eval_top_calibration, eval_top_calibration],
        eval_mse=utils.eval_top_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses)
    plot_ces(bins_list, l2_ces, l2_stddevs)


def imagenet_experiment_marginal_2_1_1000():
    logits_file = 'imagenet_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 20
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(
        data_sampler=make_calibration_data_sampler(logits, labels, num_calibration),
        num_bins_list=bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        calibration_evaluators=[eval_marginal_calibration, eval_marginal_calibration],
        eval_mse=utils.eval_marginal_mse,
        num_trials=num_trials)
    plot_mse_ce_curve(bins_list, l2_ces, mses)
    plot_ces(bins_list, l2_ces, l2_stddevs)


if __name__ == "__main__":
    cifar10_experiment_marginal_2_1_1000()
    # args = parser.parse_args()
    # logits, labels = utils.load_test_logits_labels(args.logits_file)
    # bins_list = list(range(5, 101, 5))
    # l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, args.num_bin_selection,
    #     args.num_binning, bins_list, eval_top_calibration, utils.eval_top_mse,
    #     [HistogramTopCalibrator, PlattBinnerTopCalibrator])
    # plot_mse_ce_curve(bins_list, l2_ces, mses)
    # plot_ces(bins_list, l2_ces, l2_stddevs)

