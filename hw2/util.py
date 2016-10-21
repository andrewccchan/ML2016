import numpy as np

def getTrainValidSets(data, portion) :
    dataNum = data.shape[0]
    bound = np.asscalar(np.floor(dataNum/portion))
    train = data[0:int(bound), :]
    valid = data[int(bound)+1:, :]
    # print(train)
    return (train, valid)

def normalize(data, axis) :
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    with np.errstate(divide='ignore', invalid='ignore') :
        c = np.true_divide((data - m), s)
        c[~np.isfinite(c)] = 0
    return (c, m, s)

def normalizePara(data, m, s) :
    with np.errstate(divide='ignore', invalid='ignore') :
        c = np.true_divide((data - m), s)
        c[~np.isfinite(c)] = 0
    return c

def calErr(pre, gnd) :
    pre = np.rint(pre)

    err = np.abs(pre - gnd)
    # for ct1 in range(pre.shape[0]) :
    #     print(np.asscalar(err[ct1,:]))
    err_sum = np.sum(err, axis=0)

    return float(np.asscalar(err_sum)) / pre.shape[0]

# def calErr(pre, gnd) :
#     pre = pre.T.tolist()
#
#     pre = list(map(int, pre[0]))
#     for p in pre :
#         print (pre)
#     pre = np.asarray(pre)
#     pre = np.expand_dims(pre, axis=1)
#     # pre = np.rint(pre)
#     # for ct1 in range(pre.shape[0]) :
#     #     print (str(pre[ct1,0])+','+str(gnd[ct1,0]))
#     err = np.abs(pre - gnd)
#     err_sum = np.sum(err, axis=0)
#     return float(np.asscalar(err_sum)) / pre.shape[0]

def sigmoid(X):
    return 1 / (1 + np.exp(-1*X))

def f(X, beta) :
    return sigmoid(np.dot(X, beta))

def padOne(X, dim = 0) :
    if dim == 0 :
        dataNum = X.shape[1]
        leadOnes = np.ones((1, dataNum))
        X = np.concatenate((leadOnes, X), axis=dim)

    elif dim == 1 :
        dataNum = X.shape[0]
        leadOnes = np.ones((dataNum, 1))
        X = np.concatenate((leadOnes, X), axis=dim)
    else :
        raise('Invalid dimension ', dim)

    return X
