import pickle
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt 

def load_al_unit_test(X, y) :
	# Check y values
	for ct1 in range(0, 10) :
		for ct2 in range(0, 500) :
			assert(y[ct1*500+ct2] == ct1)

	for ct1 in range(X.shape[0]) :
		assert(X[ct1].shape == (32, 32, 3))

def load_al(para) :
	# Parameters
	al_n_class = para["al_n_class"] # 0 - 9
	al_n_img_class = para["al_n_img_class"] # 0 - 499
	imgSize = para["imgSize"]
	imgChan = para["imgChan"] # number of channels for each image
	totalSize = para["totalSize"] # 1024*3

	al_X = np.zeros((al_n_class*al_n_img_class, imgSize, imgSize, imgChan))
	al_y = np.zeros((al_n_class*al_n_img_class))
	al_data = pickle.load( open("../data/all_label.p", "rb") )
	al_data = np.asarray(al_data)

	ct2 = 0 # counter of row number
	for ct1 in range(al_n_class) :
		start = ct1*al_n_img_class
		end = start + al_n_img_class
		data = np.asarray(al_data[ct1])
		tmp = np.reshape(data, (data.shape[0], \
		imgChan, imgSize, imgSize))

		# Reshape to (imgSize, imgSize, imgChan)
		al_X[start:end,:,:,	:] = np.swapaxes(tmp, 1, 3)
		al_y[start:end] = int(ct1)
	load_al_unit_test(al_X, al_y)
	al_y = np_utils.to_categorical(al_y.astype(int), al_n_class)
	print al_y[0].tolist()

	return (al_X, al_y)

def load_au(para) :
	# Parameters
	au_n_img = para["au_n_img"]
	imgChan = para["imgChan"]
	imgSize = para["imgSize"]
	totalSize = para["totalSize"]

	au_X = np.zeros((au_n_img, imgChan, imgSize, imgSize))
	au_y = np.zeros((au_n_img, 1))
	au_data = pickle.load( open("../data/all_unlabel.p", "rb") )

	au_X = np.reshape(au_data, (au_n_img, imgChan, imgSize, imgSize))

	return au_X
