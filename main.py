import cifar10vgg
from tensorflow.keras.datasets import cifar10
import numpy as np
import pickle
import bisect
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import utils

dataset = cifar10

# def get_ce_mses(logits, labels, ce_est)

def calibrate_marginals_experiment(logits, labels, k):
    num_calib = 3000
    num_bin = 3000
    num_cert = 4000
    assert(logits.shape[0] == num_calib + num_bin + num_cert)
    num_bins = 100
    bootstrap_samples = 100
    # First split by label? To ensure equal class numbers? Do this later.
    labels = utils.get_labels_one_hot(labels, k)
    mse = np.mean(np.square(labels - logits))
    print('original mse is ', mse)
    calib_logits = logits[:num_calib, :]
    calib_labels = labels[:num_calib, :]
    bin_logits = logits[num_calib:num_calib + num_bin, :]
    bin_labels = labels[num_calib:num_calib + num_bin, :]
    cert_logits = logits[num_calib + num_bin:, :]
    cert_labels = labels[num_calib + num_bin:, :]
    mses = []
    unbiased_ces = []
    biased_ces = []
    std_unbiased_ces = []
    std_biased_ces = []
    for num_bins in range(10, 21, 10):
        # Train a platt scaler and binner.
        platts = []
        platt_binners_equal_points = []
        for l in range(k):
            platt_l = utils.get_platt_scaler(calib_logits[:, l], calib_labels[:, l])
            platts.append(platt_l)
            cal_logits_l = platt_l(calib_logits[:, l])
            # bins_l = utils.get_equal_bins(cal_logits_l, num_bins=num_bins)
            # Get 
            # bins_l = utils.get_equal_prob_bins(num_bins=num_bins)
            bins_l = [0.0012, 0.05, 0.01, 0.95, 0.985, 1.0]
            cal_bin_logits_l = platt_l(bin_logits[:, l])
            platt_binner_l = utils.get_discrete_calibrator(cal_bin_logits_l, bins_l)
            platt_binners_equal_points.append(platt_binner_l)

        # Write a function that takes data and outputs the mse, ce
        def get_mse_ce(logits, labels, ce_est):
            mses = []
            ces = []
            logits = np.array(logits)
            labels = np.array(labels)
            for l in range(k):
                cal_logits_l = platt_binners_equal_points[l](platts[l](logits[:, l]))
                data = list(zip(cal_logits_l, labels[:, l]))
                bins_l = utils.get_discrete_bins(cal_logits_l)
                binned_data = utils.bin(data, bins_l)
                # probs = platts[l](logits[:, l])
                # for p in [1, 5, 10, 20, 50, 85, 88.5, 92, 94, 96, 98, 100]:
                #     print(np.percentile(probs, p))
                # import time
                # time.sleep(100)
                # print('lengths')
                # print([len(d) for d in binned_data])
                ces.append(ce_est(binned_data))
                mses.append(np.mean([(prob - label) ** 2 for prob, label in data]))
            return np.mean(mses), np.mean(ces)

        def plugin_ce_squared(data):
            logits, labels = zip(*data)
            return get_mse_ce(logits, labels, lambda x: utils.plugin_ce(x) ** 2)[1]
        def mse(data):
            logits, labels = zip(*data)
            return get_mse_ce(logits, labels, lambda x: utils.plugin_ce(x) ** 2)[0]
        def unbiased_ce_squared(data):
            logits, labels = zip(*data)
            return get_mse_ce(logits, labels, utils.improved_unbiased_square_ce)[1]

        mse, improved_unbiased_ce = get_mse_ce(
            cert_logits, cert_labels, utils.improved_unbiased_square_ce)
        mse, biased_ce = get_mse_ce(
            cert_logits, cert_labels, lambda x: utils.plugin_ce(x) ** 2)
        mses.append(mse)
        unbiased_ces.append(improved_unbiased_ce)
        biased_ces.append(biased_ce)
        print('biased ce: ', np.sqrt(biased_ce))
        print('mse: ', mse)
        print('improved ce: ', np.sqrt(improved_unbiased_ce))
        data = list(zip(list(cert_logits), list(cert_labels)))
        std_biased_ces.append(
            utils.bootstrap_std(data, plugin_ce_squared, num_samples=bootstrap_samples))
        std_unbiased_ces.append(
            utils.bootstrap_std(data, unbiased_ce_squared, num_samples=bootstrap_samples))

    std_multiplier = 1.3  # For one sided 90% confidence interval.
    upper_unbiased_ces = list(map(lambda p: np.sqrt(p[0] + std_multiplier * p[1]),
                                  zip(unbiased_ces, std_unbiased_ces)))
    upper_biased_ces = list(map(lambda p: np.sqrt(p[0] + std_multiplier * p[1]),
                                zip(biased_ces, std_biased_ces)))
    # Get points on the Pareto curve, and plot them.
    def get_pareto_points(data):
        pareto_points = []
        def dominated(p1, p2):
            return p1[0] >= p2[0] and p1[1] >= p2[1]
        for datum in data:
            num_dominated = sum(map(lambda x: dominated(datum, x), data))
            if num_dominated == 1:
                pareto_points.append(datum)
        return pareto_points
    print(get_pareto_points(list(zip(upper_unbiased_ces, mses, list(range(5, 101, 5))))))
    print(get_pareto_points(list(zip(upper_biased_ces, mses, list(range(5, 101, 5))))))
    plot_unbiased_ces, plot_unbiased_mses = zip(*get_pareto_points(list(zip(upper_unbiased_ces, mses))))
    plot_biased_ces, plot_biased_mses = zip(*get_pareto_points(list(zip(upper_biased_ces, mses))))
    plt.title("MSE vs Calibration Error")
    plt.scatter(plot_unbiased_ces, plot_unbiased_mses, c='red', marker='o', label='Ours')
    plt.scatter(plot_biased_ces, plot_biased_mses, c='blue', marker='s', label='Plugin')
    plt.legend(loc='upper left')
    plt.ylim(0.0, 0.013)
    plt.xlabel("L2 Calibration Error")
    plt.ylabel("Mean-Squared Error")
    plt.show()


if __name__ == "__main__":
    # save_cifar_validation_logits(cifar10, cifar10vgg.cifar10vgg(), "logits.dat")
    # save_cifar_validation_preds(cifar10, cifar10vgg.cifar10vgg(), "predictions.dat")
    logits = pickle.load(open("logits.dat", "rb"))
    (_, _), (_, y_test) = dataset.load_data()
    predictions = np.argmax(logits, 1)
    probabilities = np.max(logits, 1)
    accuracy = sum(y_test[:, 0] == predictions)
    print('accuracy is ' + str(accuracy))

    calibrate_marginals_experiment(logits, y_test, k=10)
    # predictions, probabilities = pickle.load(open("predictions.dat", "rb"))


    # # Compute validation and test preds and probabilities.
    # num_valid = 2000
    # valid_preds = predictions[:num_valid]
    # valid_probs = probabilities[:num_valid]
    # valid_correct = (y_test[:num_valid, 0] == valid_preds)
    # valid_data = list(zip(valid_probs, valid_correct))
    # test_preds = predictions[num_valid:]
    # test_probs = probabilities[num_valid:]
    # test_correct = (y_test[num_valid:, 0] == test_preds)
    # test_data = list(zip(test_probs, test_correct))
    # print(len(test_correct))

    # # accuracy = sum(correct) * 1.0 / correct.shape[0]
    # # print(accuracy)
    # # Uncalibrated.
    # for i in [4, 8, 16]:
    #     bins = utils.get_equal_bins(test_probs, num_bins=i)
    #     binned_data = utils.bin(test_data, bins)
    #     print(utils.unbiased_l2_ce(binned_data))


    # # Calibrate on validation set.
    # # calibrator = naive_bin_calibrator(valid_probs, valid_correct, bins=128)
    # # # calibrator = naive_bin_calibrator(valid_probs, valid_correct)
    # # bin_test_probs = map(calibrator, test_probs)
    # # print(set(bin_test_probs))

    # # Fit logistic model on calibration set.
    # # clf = LogisticRegression(C=1e5, solver='lbfgs')
    # # valid_probs = np.expand_dims(valid_probs, axis=-1)
    # # valid_probs = np.log(valid_probs / (1 - valid_probs))
    # # clf.fit(valid_probs, valid_correct)
    # # plt.figure(1, figsize=(4, 3))
    # # plt.clf()
    # # plt.scatter(valid_probs.ravel(), valid_correct, color='black', zorder=20)

    # # # # Make predictions on test set. 
    # # def model(x):
    # #     return 1 / (1 + np.exp(-x))
    # # test_probs = np.log(test_probs / (1 - test_probs))
    # # cal_test_probs = model(test_probs * clf.coef_ + clf.intercept_).ravel()

    # for i in [3, 10, 16]:
    #     platt = utils.get_platt_scaler(valid_probs, valid_correct)
    #     cal_valid_probs = platt(valid_probs)
    #     bins = utils.get_equal_bins(cal_valid_probs, num_bins=i)
    #     platt_binner = utils.get_discrete_calibrator(cal_valid_probs, bins)
    #     cal_test_probs = platt_binner(platt(test_probs))
    #     test_data = list(zip(cal_test_probs, test_correct))

    #     # bins = [1.0/8, 2.0/8, 3.0/8, 4.0/8, 5.0/8, 6.0/8, 7.0/8, 1.0]
    #     # bins = utils.get_equal_bins(cal_test_probs, num_bins=i)
    #     bins = utils.get_discrete_bins(cal_test_probs)
    #     def estimator(data):
    #         binned_data = utils.bin(data, bins)
    #         return utils.unbiased_square_ce(binned_data)
    #     def plugin_estimator(data):
    #         binned_data = utils.bin(data, bins)
    #         return utils.plugin_ce(binned_data)
    #     # print('estimate for %d: %f' % (i, estimator(test_data)))
    #     # print('unbiased uncertainty: ',
    #     #       utils.bootstrap_uncertainty(test_data, plugin_estimator, estimator))
    #     print('plugin estimate for %d: %f' % (i, plugin_estimator(test_data)))
    #     print('plugin uncertainty: ', utils.bootstrap_uncertainty(test_data, plugin_estimator))

    #     # Measure MSE.
    #     mse = np.mean([(prob - label) ** 2 for prob, label in test_data])
    #     print(mse)


    # # X_test = np.linspace(0, 1, 300)
    # # loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
    # # plt.plot(X_test, loss, color='red', linewidth=3)
    # # plt.show()

    # # for i in [1, 2, 4, 8, 16, 32, 64, 128]:
    # #   ave_probs, ave_correct, counts = get_stats(bin_test_probs, test_correct, bins=i)
    # #   # print(ave_probs, ave_correct, counts)
    # #   print(ece(counts, ave_probs, ave_correct))











