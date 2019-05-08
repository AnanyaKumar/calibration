
import argparse
import calibrators
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


import utils

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')
parser.add_argument('--num_bin_selection', default=500, type=int,
                    help='Number of examples to use for Platt Scaling.')
parser.add_argument('--num_binning', default=500, type=int,
                    help='Number of examples to use for binning.')


def eval_top_calibration(probs, logits, labels, plugin=True):
    correct = (utils.get_top_predictions(logits) == labels)
    data = list(zip(probs, correct))
    bins = utils.get_discrete_bins(probs)
    binned_data = utils.bin(data, bins)
    if plugin:
        return utils.plugin_ce(binned_data) ** 2
    else:
        return utils.improved_unbiased_square_ce(binned_data)
    # print([np.mean(np.array(l)[:, 0]) for l in binned_data])
    # print([np.mean(np.array(l)[:, 1]) for l in binned_data])
    # print(utils.plugin_ce(binned_data, power=1))
    # def estimator(data):
    #   binned_data = utils.bin(data, bins)
    #   return utils.plugin_ce(binned_data, power=2)
    # return utils.bootstrap_uncertainty(data, estimator, num_samples=num_samples)


def eval_top_mse(probs, logits, labels):
    correct = (utils.get_top_predictions(logits) == labels)
    return np.mean(np.square(probs - correct))


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
        ce_c = utils.plugin_ce(binned_data_c) ** 2
        ces.append(ce_c)
    return np.mean(ces)


def eval_marginal_mse(probs, logits, labels):
    assert probs.shape == logits.shape
    return np.mean(np.square(probs - logits))


def compare_calibrators(logits, labels, num_calibration, num_bins, Calibrators,
                        eval_calibration, eval_mse, resample=True):
    assert len(logits) == len(labels)
    if resample:
        indices = np.random.choice(list(range(len(logits))),
                                   size=num_calibration, replace=True)
    else:
        indices = np.array(list(range(len(logits))))
        np.random.shuffle(indices)
    shuffled_logits = [logits[i] for i in indices]
    shuffled_labels = [labels[i] for i in indices]
    train_logits = shuffled_logits[:num_calibration]
    train_labels = shuffled_labels[:num_calibration]
    if resample:
        eval_logits = logits
        eval_labels = labels
    else:
        eval_logits = shuffled_logits[num_calibration:]
        eval_labels = shuffled_labels[num_calibration:]
    l2_ces = []
    mses = []
    for Calibrator in Calibrators:
        calibrator = Calibrator(num_calibration, num_bins)
        calibrator.train_calibration(train_logits, train_labels)
        calibrated_probs = calibrator.calibrate(eval_logits)
        mid = eval_calibration(calibrated_probs, eval_logits, eval_labels, plugin=resample)
        mse = eval_mse(calibrated_probs, eval_logits, eval_labels)
        l2_ces.append(mid)
        mses.append(mse)
    return l2_ces, mses


def average_calibration(logits, labels, num_calibration, num_bins, Calibrators,
                        eval_calibration, eval_mse, num_trials=100, resample=True):
    l2_ces, mses = [], []
    for _ in range(num_trials):
        cur_l2_ces, cur_mses = compare_calibrators(
            logits, labels, num_calibration, num_bins, Calibrators, eval_calibration, eval_mse,
            resample=resample)
        l2_ces.append(cur_l2_ces)
        mses.append(cur_mses)
    l2_ce_means = np.mean(l2_ces, axis=0)
    l2_ce_stddevs = np.std(l2_ces, axis=0) / np.sqrt(num_trials)
    mses = np.mean(mses, axis=0)
    mse_stddevs = np.std(mses, axis=0) / np.sqrt(num_trials)
    return l2_ce_means, l2_ce_stddevs, mses, mse_stddevs


def vary_bin_calibration(logits, labels, num_calibration, num_bins_list,
                         Calibrators, eval_calibration, eval_mse, num_trials=100, resample=True):
    ce_list = []
    stddev_list = []
    mse_list = []
    for num_bins in num_bins_list:
        l2_ce_means, l2_ce_stddevs, mses, mse_stddevs = average_calibration(
            logits, labels, num_calibration, num_bins, Calibrators,
            eval_calibration, eval_mse, num_trials, resample=resample)
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

    plt.show()


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
    plt.show()


def cifar10_experiment_top_1_1_1000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, num_calibration,
        bins_list,
        Calibrators=[calibrators.HistogramTopCalibrator, calibrators.PlattBinnerTopCalibrator],
        eval_calibration=eval_top_calibration, eval_mse=eval_top_mse, num_trials=num_trials,
        resample=True)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.002), ylim=(0.0425, 0.045))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_top_1_2_3000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 3000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, num_calibration,
        bins_list,
        Calibrators=[calibrators.HistogramTopCalibrator, calibrators.PlattBinnerTopCalibrator],
        eval_calibration=eval_top_calibration, eval_mse=eval_top_mse, num_trials=num_trials,
        resample=True)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.002), ylim=(0.0425, 0.045))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_marginal_2_1_1000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 100
    num_calibration = 1000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, num_calibration,
        bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        eval_calibration=eval_marginal_calibration, eval_mse=eval_marginal_mse,
        num_trials=num_trials, resample=True)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.001), ylim=(0.0, 0.0075))
    plot_ces(bins_list, l2_ces, l2_stddevs)


def cifar10_experiment_marginal_2_2_3000():
    logits_file = 'cifar_logits.dat'
    logits, labels = utils.load_test_logits_labels(logits_file)
    bins_list = list(range(10, 101, 10))
    num_trials = 20
    num_calibration = 3000
    l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, num_calibration,
        bins_list,
        Calibrators=[calibrators.HistogramMarginalCalibrator,
                     calibrators.PlattBinnerMarginalCalibrator],
        eval_calibration=eval_marginal_calibration, eval_mse=eval_marginal_mse,
        num_trials=num_trials, resample=True)
    plot_mse_ce_curve(bins_list, l2_ces, mses, xlim=(0.0, 0.001), ylim=(0.0, 0.0075))
    plot_ces(bins_list, l2_ces, l2_stddevs)


if __name__ == "__main__":
    cifar10_experiment_marginal_2_2_3000()
    # args = parser.parse_args()
    # logits, labels = utils.load_test_logits_labels(args.logits_file)
    # bins_list = list(range(5, 101, 5))
    # l2_ces, l2_stddevs, mses = vary_bin_calibration(logits, labels, args.num_bin_selection,
    #     args.num_binning, bins_list, eval_top_calibration, eval_top_mse,
    #     [HistogramTopCalibrator, PlattBinnerTopCalibrator])
    # plot_mse_ce_curve(bins_list, l2_ces, mses)
    # plot_ces(bins_list, l2_ces, l2_stddevs)

