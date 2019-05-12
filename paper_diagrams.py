import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import utils
from matplotlib import rc

_PLATT_SCALING = 0
_HIST_BINNING = 1
_VAR_RED_BINNING = 2
calibrator = _HIST_BINNING

if __name__ == "__main__":
	font = {'size': 14}
	rc('font', **font)

	plt.figure(figsize=(4,3))
	plt.yticks([0, 1])
	plt.xticks([0, 0.5, 1])
	ax = plt.gca()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.set_ylim([-0.05, 1.05])

	# Set up points.
	X = np.arange(1.0 / 18, 1.0, 1.0 / 9)
	Y = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
	plt.scatter(X, Y, marker='x', s=40, c='black')

	# Add Platt curve.
	platt_scaler = utils.get_platt_scaler(X, Y)
	finer_x = np.arange(0.0, 1.0, 0.01)
	platt_finer_y = platt_scaler(finer_x)
	if calibrator == _PLATT_SCALING:
		plt.plot(finer_x, platt_finer_y, c='red')
	elif calibrator == _VAR_RED_BINNING:
		plt.plot(finer_x, platt_finer_y, c='gray')

	# Add bin lines.
	if calibrator == _VAR_RED_BINNING or calibrator == _HIST_BINNING:
		plt.axvline(x=1/3.0, linestyle='--', linewidth=2, c='gray')
		plt.axvline(x=2/3.0, linestyle='--', linewidth=2, c='gray')

	# Add platt values.
	platt_y = platt_scaler(X)
	if calibrator == _VAR_RED_BINNING:
		plt.scatter(X, platt_y, marker='o', s=40, c='gray')

	# Add average lines.
	if calibrator == _HIST_BINNING:
		outputs = Y
	else:
		outputs = platt_y
	output1 = np.mean(outputs[:3])
	output2 = np.mean(outputs[3:6])
	output3 = np.mean(outputs[6:])
	averages = [output1, output2, output3]
	eps = 1e-9
	if calibrator == _HIST_BINNING or calibrator == _VAR_RED_BINNING:
		plt.plot([0.0, 1/3.0], [averages[0], averages[0]], c='red')
		plt.plot([1/3.0, 2/3.0], [averages[1], averages[1]], c='red')
		plt.plot([2/3.0, 3/3.0], [averages[2], averages[2]], c='red')
		for i in range(3):
			for j in range(3):
				plt.plot([X[3*i+j], X[3*i+j]], [averages[i],outputs[3*i+j]], c='blue',
					     linewidth=2, linestyle=':')

	if calibrator == _PLATT_SCALING or calibrator == _VAR_RED_BINNING:
		plt.xlabel(" ")
	else:
		plt.xlabel("Uncalibrated Model Output")
	if calibrator == _PLATT_SCALING:
		plt.ylabel("Calibrator Output")
	plt.tight_layout()

	plt.show()