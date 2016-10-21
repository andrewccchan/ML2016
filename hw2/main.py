from train import train_linear

from parseData import parseData
from eva import evaluate
from util import getTrainValidSets, normalize, normalizePara
import numpy as np
import sys

# print ('Reading training data')
[feature, target] = parseData('./data/spam_train.csv')

# Get training and validation sets
(train_fea, test_fea) = getTrainValidSets(feature, 2)
(train_tar, test_tar) = getTrainValidSets(target, 2)

train_fea = feature
train_tar = target

# Feature normalization
(train_fea, m, s) = normalize(train_fea, axis=0)
test_fea = normalizePara(test_fea, m, s)

# print ('Training')
# Training
beta = train_linear(train_fea, train_tar, test_fea, test_tar, eta=1e-5, lamb=0, maxIter=1000000, debug=0)
# print(factors)
# evaluate('./data/spam_test.csv', beta, sys.argv[1])
evaluate('./data/spam_test.csv', beta, 'submit', m, s)
