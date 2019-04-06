
import utils
import numpy as np


def well_balanced(bins, alpha):
	B = len(bins)
	bins = [0.0] + bins
	for i in range(B - 1):
		width = bins[i+1] - bins[i]
		if width > 1.0 * alpha / B:
			return False
		elif width < 1.0 / B / alpha:
			return False
	return True

def test_binning(B, N, T):
	num_well_balanced = 0
	for i in range(T):
		samples = np.random.uniform(size=N)
		bins = utils.get_equal_bins(samples, num_bins=B)
		num_well_balanced += well_balanced(bins, 2.0)
	return 1.0 * num_well_balanced / T

for B in [10, 30, 100, 200, 300, 1000, 5000, 10000]:
	N = int(6 * B * np.log(B))
	print(test_binning(B, N, 50))
