import numpy as np

def rmse(diff) :
    dataNum = diff.shape[0]
    # summ = np.linalg.norm(diff, 2)
    summ = np.square(diff)
    summ = np.average(summ, axis=0)
    return np.sqrt(summ)

def train_linear(trainData, gndData, testTrain, testGnd, eta, lamb, tol, maxIter, debug) :
    dataNum = gndData.shape[0]
    dimNum = trainData.shape[1]
    beta = np.zeros([dimNum, 1])

    converged = False
    curIter = 0

    while not converged :
        curIter = curIter + 1
        tmp = np.dot(trainData, beta) - gndData
        new_beta = beta - 2*eta*np.dot(np.transpose(trainData), tmp) + lamb * beta
        diff = np.linalg.norm(new_beta - beta, 2)
        beta = new_beta

        if debug == 1 :
            print('Iter: ', curIter, '\t change: ', diff)

        # evaluate training resutls
        if curIter % 1000 == 0 :
            trainErr = np.dot(trainData, beta) - gndData
            testErr = np.dot(testTrain, beta) - testGnd

            print('Iter: ', curIter, '\n Train Err: ', rmse(trainErr), '\t Train std: ', np.std(trainErr), '\n Test Err: ', rmse(testErr), '\t Test std: ', np.std(testErr))

        if curIter > maxIter :
            converged = True



    return beta
