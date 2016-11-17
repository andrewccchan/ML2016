import numpy as np
import pickle
from util import parsePara, getTrainValidSet
from os.path import isfile
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import Sequential
import keras
import sys
import os

# Load data
para = parsePara();

# Get .p file path
al_path = os.path.join(sys.argv[1], "all_label.p")
un_path = os.path.join(sys.argv[1], "all_unlabel.p")
test_path = os.path.join(sys.argv[1], "test.p")
# Load labled data
print "Loading labeled data"
if (not isfile("al_X.npy")) :
    al_data = pickle.load( open(al_path, "rb") )
    al_X = np.concatenate([a for a in al_data])
    al_y = []
    for ct1 in range(10) :
        al_y.extend([ct1] * 500)
    al_X = np.asarray(al_X)
    # al_X = np.reshape(al_X, (5000, 3, 32, 32))
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
    un_X = np.asarray(un_X)
    # un_X = np.reshape(un_X, (45000, 3, 32, 32))
    np.save("un_X_in", un_X)
else :
    un_X = np.load("un_X_in.npy")

print "Loading test data"
if (not isfile("test_X.npy")) :
    test_X = pickle.load( open(test_path, "rb") )
    test_X = np.asarray(test_X["data"])
    np.save("test_X", test_X)
else :
    test_X = np.load("test_X.npy")

al_X = al_X.astype("float32")
al_X /= 255
un_X = un_X.astype("float32")
un_X /= 255
test_X = test_X.astype("float32")
test_X /= 255
# trans = para["transductive"]
# if (trans) :
#     test = pickle.load( open("../data/test.p", "rb") )
#     test = np.reshape(test, (10000, 3, 32, 32))
#     au_X = np.concatenate((au_X, test), 0)

batch_size = para["batch_size"]

# Deep autoencoder
input_layer = Input(shape=(3072,))
hidLay1 = Dense(1024, activation='relu')(input_layer)
hidLay2 = Dense(512, activation='relu')(hidLay1)
code = Dense(256, activation='relu')(hidLay2)
hidLay3 = Dense(512, activation='relu')(code)
hidLay4 = Dense(1024, activation='relu')(hidLay3)
output_layer = Dense(3072, activation='relu')(hidLay4)

autoencoder = Model(input=input_layer,
                    output=output_layer)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

X = np.concatenate((al_X, un_X), axis=0)
# (al_X, va_X, al_y, va_y) = getTrainValidSet(al_X, al_y, 0.1)
print "Fitting autoencoder"
autoencoder.fit(X, X, nb_epoch=10,
                batch_size=batch_size,
                shuffle=True)

# Encoding network
enIn = Input(shape=(3072,))
enLay1 = Dense(1024, activation='relu',
        weights=autoencoder.layers[1].get_weights())(enIn)
enLay2 = Dense(512, activation='relu',
        weights=autoencoder.layers[2].get_weights())(enLay1)
enCode = Dense(256, activation='relu',
        weights=autoencoder.layers[3].get_weights())(enLay2)

enNet = Model(input=enIn, output=enCode)

print "Encoding al_X and un_X"
al_X_en = enNet.predict(al_X, verbose=1)
assert(al_X_en.shape[0] == al_X.shape[0])
assert(al_X_en.shape[1] == 256)
un_X_en = enNet.predict(un_X, verbose=1)
assert(un_X_en.shape[0] == un_X.shape[0])
assert(un_X_en.shape[1] == 256)

# Classificaiton NN
clsfIn = Input(shape=(256,))
clsfLay1 = Dense(1024, activation='relu')(clsfIn)
clsfOut = Dense(10, activation='sigmoid')(clsfLay1)

clsf = Model(input=clsfIn, output=clsfOut)
clsf.compile(optimizer="adam", loss="binary_crossentropy")


# self-training
print "Training classifier"
clsf.fit(al_X_en, al_y,
        batch_size=batch_size,
        nb_epoch=10,
        shuffle=True)

for ct1 in range(15) :
    print "Retraining FCN: iter=", str(ct1)
    if un_X_en.size == 0 :
        print "un_X is empty. Training stops"
        break
    else :
        print "Training data size: ", str(al_X_en.shape[0])
        pre = clsf.predict(un_X_en, verbose=1)
        maskSe = np.squeeze((np.amax(pre, 1) > 0.9))
        maskRm = np.logical_not(maskSe) # Rm stands for remaining
        newLab = pre[maskSe,...]
        newLab = (newLab == newLab.max(axis=1)[:,None]).astype(int)

        # Add to labeled data and remove form un_X
        al_X_en = np.append(al_X_en, un_X_en[maskSe,...], axis=0)
        al_y = np.append(al_y, newLab, axis=0)
        un_X_en = un_X_en[maskRm]

        clsf.fit(al_X_en, al_y,
                batch_size=batch_size,
                nb_epoch=10,
                shuffle=True)

#from keras.models import model_from_json
#enNet_js = enNet.to_json()
#clsf_js = clsf.to_json()
#with open (sys.argv[2]+"_en.json", "w") as js_file:
#    js_file.write(enNet_js) 
#with open(sys.argv[2]+"_cl.json", "w") as js_file2:
#    js_file2.write(clsf_js)
enNet.save(sys.argv[2]+"_en.h5")
clsf.save(sys.argv[2]+"_cl.h5")

# predict
# test_en = enNet.predict(test_X)
# pre = clsf.predict(test_en, verbose=1)
# pre = (pre == pre.max(axis=1)[:,None]).astype(int)
#
# with open("submit_method2.csv", "w") as f :
#     header = ["ID", "class"]
#     f.write(",".join(header) + "\n")
#
#     for ct1 in range(pre.shape[0]) :
#         p = np.argmax(pre[ct1])
#         f.write(",".join([str(ct1), str(p)]) + "\n")
