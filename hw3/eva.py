import numpy as np
import pickle
import sys
import os
import keras
from util import parsePara

test_path = os.path.join(sys.argv[1], "test.p")
enNet = keras.models.load_model(sys.argv[2]+"_en.h5")
clsf = keras.models.load_model(sys.argv[2]+"_cl.h5")

para = parsePara()
imgChan = para["imgChan"]
imgSize= para["imgSize"]
batch_size = para["batch_size"]
test_n_img = para["test_n_img"]

test_data = pickle.load( open(test_path, "rb") )
test_id = test_data["ID"]
test = np.asarray(test_data["data"])

test = test.astype("float32")
test /= 255

test_en = enNet.predict(test)
pre = clsf.predict(test_en, verbose=1)
pre = (pre == pre.max(axis=1)[:,None]).astype(int)

with open(sys.argv[3], "w") as f :
    header = ["ID", "class"]
    f.write(",".join(header) + "\n")

    for ct1 in range(pre.shape[0]) :
        p = np.argmax(pre[ct1])
        f.write(",".join([str(ct1), str(p)]) + "\n")
