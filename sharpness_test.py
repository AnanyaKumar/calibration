
import numpy as np

p_1 = 0.6
p_2 = 0.9
p_ave = (p_1 + p_2) / 2.0

def abs_error(true_prob, pred_prob):
	return true_prob * abs(1 - pred_prob) + (1 - true_prob) * abs(pred_prob)

def mse(true_prob, pred_prob):
	return true_prob * ((1 - pred_prob) ** 2) + (1 - true_prob) * (pred_prob ** 2)

def log_likelihood(true_prob, pred_prob):
	return true_prob * np.log(pred_prob) + (1 - true_prob) * np.log(1 - pred_prob)

finer_abs = 0.5 * abs_error(p_1, p_1) + 0.5 * abs_error(p_2, p_2)
worse_abs = 0.5 * abs_error(p_1, p_ave) + 0.5 * abs_error(p_2, p_ave)
print(finer_abs, worse_abs, worse_abs / finer_abs)

finer_mse = 0.5 * mse(p_1, p_1) + 0.5 * mse(p_2, p_2)
worse_mse = 0.5 * mse(p_1, p_ave) + 0.5 * mse(p_2, p_ave)
print(finer_mse, worse_mse, worse_mse / finer_mse)

finer_ll = 0.5 * log_likelihood(p_1, p_1) + 0.5 * log_likelihood(p_2, p_2)
finer_l = np.exp(finer_ll)
worse_ll = 0.5 * log_likelihood(p_1, p_ave) + 0.5 * log_likelihood(p_2, p_ave)
worse_l = np.exp(worse_ll)
print(finer_ll, worse_ll, worse_ll / finer_ll )
print(finer_l, worse_l, finer_l / worse_l)
