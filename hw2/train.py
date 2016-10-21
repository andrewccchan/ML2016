import numpy as np
from util import calErr, sigmoid, f, padOne

def train_linear(train_fea, train_tar, test_fea, test_tar, eta, lamb, maxIter, debug) :
    dataNum = train_fea.shape[0]
    dimNum = train_fea.shape[1]
    beta = np.zeros([dimNum+1, 1])
    # beta = np.zeros([dimNum, 1])

    # Pad leading 1s to the beginging of training data
    train_fea = padOne(train_fea, 1)
    test_fea = padOne(test_fea, 1)

    converged = False
    curIter = 0

    while not converged :
        curIter = curIter + 1
        tmp = f(train_fea, beta) - train_tar
        new_beta = beta - 2*eta*np.dot(np.transpose(train_fea), tmp) + lamb * beta
        diff = np.linalg.norm(new_beta - beta, 2)
        beta = new_beta

        if debug == 1 :
            print('Iter: ', curIter, '\t change: ', diff)

        # evaluate training resutls
        if curIter % 1000 == 0 :
            train_res = f(train_fea, beta)
            test_res = f(test_fea, beta)
            # calErr(test_res, test_tar)
            print('Iter: ', curIter, '\t Train Err: ', calErr(train_res, train_tar), '\t Test Err: ', calErr(test_res, test_tar))

        if curIter > maxIter :
            converged = True

    # train_res = f(train_fea, beta)
    # for ct1 in range(test_fea.shape[0]) :
    #     print (test_fea[ct1,:])
    # test_res = f(test_fea, beta)
    # print(calErr(test_res, test_tar))

    return beta
