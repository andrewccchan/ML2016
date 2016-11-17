import numpy as np


def getTV_unit_test(shufIdx, X, y, Xp, yp) :
	idx = np.argwhere(shufIdx == 100)
	idx = np.asscalar(idx)
	assert(np.array_equal(X[idx], Xp[100]))
	assert(np.array_equal(y[idx], yp[100]))

def getTrainValidSet(X, y, r) :
    # X should have size [dataNumximgSizeximgSizeximgChan]
    # y should have size [dataNumxnb_class]
    assert(X.shape[0] == y.shape[0])
    # shufIdx = np.random.permutation(X.shape[0])
    shufIdx = range(X.shape[0])
    np.random.shuffle(shufIdx)
    Xp = X
    yp = y
    X = X[shufIdx]
    y = y[shufIdx]
    # getTV_unit_test(shufIdx, X, y, Xp, yp)

    mid= int(np.floor(X.shape[0]*(1.-r)))
    # Return values: X_trian, X_valid, y_train, y_valid
    return (X[0:mid], X[mid:], y[0:mid], y[mid:])

def parsePara(paraFile="./paras") :
	para = dict()
	with open("./paras", "r") as pf :
		for l in pf :
			# Remove white space
			l = l.replace(" ", "")
			# Remove comments
			cont = l.split("#")

			if cont[0] != "" :
				contArr = cont[0].split("=")
				num = contArr[1]
				num = float(num) if "." in num else int(num)
				para[contArr[0]] = num

	return para
