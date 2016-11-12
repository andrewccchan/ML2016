from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def buildCNN(inputSize, para) :
	m = para["filterM"]
	n	= para["filterN"]
	filNum = para["filterNum"]
	drop = para["drop"]
	nb_classes = para["al_n_class"]
	model = Sequential()


	model.add(Convolution2D(filNum, m, n, border_mode='same',
	                        input_shape=inputSize))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, m, n))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(drop))

	model.add(Convolution2D(64, m, n, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, m, n))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(drop))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(drop))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model
