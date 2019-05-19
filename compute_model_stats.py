
import argparse
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--logits_file', default='cifar_logits.dat', type=str,
                    help='Name of file to load logits, labels pair.')


if __name__ == "__main__":
	args = parser.parse_args()
	logits, labels = utils.load_test_logits_labels(args.logits_file)
	# Get prediction accuracy.
	predictions = utils.get_top_predictions(logits)
	probs = utils.get_top_probs(logits)
	correct = (predictions == labels)
	print('accuracy: ', float(sum(correct)) / len(labels))
	# Get top-label MSE.
	top_mse = utils.eval_top_mse(probs, logits, labels)
	print('top mse: ', top_mse)
	# Get marginal MSE.
	marginal_mse = utils.eval_marginal_mse(logits, logits, labels)
	print('marginal mse: ', marginal_mse)
