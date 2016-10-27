import numpy as np
from ann_new import fit

def train_ann(X, Y, testX, testY) :
	return fit(X, Y, 128, testX, testY, 50, 5e-5, 0, 700)
