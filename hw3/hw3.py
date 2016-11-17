import numpy as np
import pickle
from util import parsePara, getTrainValidSet
from os.path import isfile
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import sys
import os
# Load data
para = parsePara();

# Get .p file path
al_path = os.path.join(sys.argv[1], "all_label.p")
un_path = os.path.join(sys.argv[1], "all_unlabel.p")
# Load labled data
print "Loading labeled data"
if (not isfile("al_X.npy")) :
    al_data = pickle.load( open(al_path, "rb") )
    al_X = np.concatenate([a for a in al_data])
    al_y = []
    for ct1 in range(10) :
        al_y.extend([ct1] * 500)
    al_X = np.reshape(al_X, (5000, 3, 32, 32))
    al_y = np_utils.to_categorical(al_y, 10)
    np.save("al_X", al_X)
    np.save("al_y", al_y)
else :
    al_X = np.load("al_X.npy")
    al_y = np.load("al_y.npy")

# Load unlabled data
print "Loading unlabeled data"
if (not isfile("un_X_in.npy")) :
    un_X = pickle.load( open(un_path, "rb") )
    un_X = np.reshape(un_X, (45000, 3, 32, 32))
    np.save("un_X_in", un_X)
else :
    un_X = np.load("un_X_in.npy")
# trans = para["transductive"]
# if (trans) :
#     test = pickle.load( open("../data/test.p", "rb") )
#     test = np.reshape(test, (10000, 3, 32, 32))
#     au_X = np.concatenate((au_X, test), 0)

# Build CNN model
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=al_X.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', \
    optimizer="adam", metrics=['accuracy'])

# Training
print "CNN self training"
batch_size = para["batch_size"]
max_iter = para["self_max_iter"]
prob_thresh = para["prob_thresh"]
first_epoch = para["basic_nb_epoch"]
self_epoch = para["self_nb_epoch"]

al_X = al_X.astype("float32")
un_X = un_X.astype("float32")
al_X /= 255
un_X /= 255
# Split training and validation data
print "Training on labeled data"
(al_X, va_X, al_y, va_y) = getTrainValidSet(al_X, al_y, 0.1)
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

print "Fitting datagen"
datagen.fit(al_X)
model.fit_generator(datagen.flow(al_X, al_y,
        batch_size=batch_size),
        samples_per_epoch=al_X.shape[0],
        nb_epoch=first_epoch,
        validation_data=(va_X, va_y))

for ct1 in range(max_iter) :
    print "Retraining: iter=", str(ct1)
    if un_X.size == 0 :
        print "un_X is empty. Training stops"
        break
    else :
        print "Training data size: ", str(al_X.shape[0])
    pre = model.predict(un_X, 1)
    maskSe = np.squeeze((np.amax(pre, 1) > prob_thresh))
    maskRm = np.logical_not(maskSe) # Rm stands for remaining
    newLab = pre[maskSe,...]
    newLab = (newLab == newLab.max(axis=1)[:,None]).astype(int)

    # Add to labeled data and remove form un_X
    al_X = np.append(al_X, un_X[maskSe,...], axis=0)
    al_y = np.append(al_y, newLab, axis=0)
    un_X = un_X[maskRm]

    #earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    
    model.fit(al_X, al_y,
             batch_size=batch_size,
             nb_epoch=self_epoch,
             validation_data=(va_X, va_y),
             shuffle=True)
             #callbacks=[earlyStopping])

model.save(sys.argv[2]+".h5")
# Prediction on public set
# predict(model, para)
