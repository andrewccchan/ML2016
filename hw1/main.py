from train import train_linear

from parseData import parseData, parseData_sliding
from eva import evaluate
from util import getTrainValidSets, normalize
import numpy as np

[history, target, factors, trainLen] = parseData_sliding('./data/train.csv')

# Get training and validation sets
(trainData, testTrain) = getTrainValidSets(history, 1.1)
(gndData, testGnd) = getTrainValidSets(target, 1.1)
#
trainData = history
gndData = target
# Feature normalization
# trainData = normalize(trainData, axis=0)
# testTrain = normalize(testTrain, axis=0)
# trainData = np.divide(trainData, 2)
# testTrain = np.divide(testTrain, 2)

beta = train_linear(trainData, gndData, testTrain, testGnd, 5e-9, 1e-10, 1e-6, 500000, 0)
print(factors)
evaluate('./data/test_X.csv', beta, factors, trainLen)
