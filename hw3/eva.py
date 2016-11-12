import numpy as np
import pickle

def eva(model, X, y, para) :
    imgChan = para["imgChan"]
    imgSize= para["imgSize"]
    batch_size = para["batch_size"]
    test_n_img = para["test_n_img"]

    pre = model.predict_classes(X, batch_size, 1)
    errCt = 0
    with open("eva.csv", "w") as f :
        header = ["pre", "gnd"]
        f.write(",".join(header) + "\n")
        ct1 = 0
        for p in pre :
            yIdx = np.asscalar(np.argwhere(y[ct1] != 0))
            errCt += 1 if (p != yIdx) else 0
            f.write(",".join([str(p), str(yIdx)]) + "\n")
            ct1 += 1
    print "Evaluation Error: ", float(errCt) / ct1

def predict(model, para) :
    imgChan = para["imgChan"]
    imgSize= para["imgSize"]
    batch_size = para["batch_size"]
    test_n_img = para["test_n_img"]

    test = np.zeros((test_n_img, imgSize, imgSize, imgChan))
    test_data = pickle.load( open("../data/test.p", "rb") )
    test_id = test_data["ID"]
    test_data = np.asarray(test_data["data"])
    test = np.reshape(test_data, (test_n_img, imgChan, imgSize, imgSize))
    test = np.swapaxes(test, 1, 3)

    test = test.astype("float32")
    test /= 255

    pre = model.predict_classes(test, batch_size, 1)

    with open("submit.csv", "w") as f :
        header = ["ID", "class"]
        f.write(",".join(header) + "\n")
        ct1 = 0
        for p in pre :
            f.write(",".join([str(ct1), str(p)]) + "\n")
            ct1 += 1
