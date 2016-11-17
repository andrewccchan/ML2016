import numpy as np
import pickle
import sys
import os
import keras
from util import parsePara

test_path = os.path.join(sys.argv[1], "test.p")
model = keras.models.load_model(sys.argv[2]+".h5")
para = parsePara() 
imgChan = para["imgChan"]
imgSize= para["imgSize"]
batch_size = para["batch_size"]
test_n_img = para["test_n_img"]

test = np.zeros((test_n_img, imgSize, imgSize, imgChan))
test_data = pickle.load( open(test_path, "rb") )
test_id = test_data["ID"]
test_data = np.asarray(test_data["data"])
test = np.reshape(test_data, (test_n_img, imgChan, imgSize, imgSize))

test = test.astype("float32")
test /= 255

pre = model.predict_classes(test, batch_size, 1)

with open(sys.argv[3], "w") as f :
    header = ["ID", "class"]
    f.write(",".join(header) + "\n")
    ct1 = 0
    for p in pre :
        f.write(",".join([str(ct1), str(p)]) + "\n")
        ct1 += 1
