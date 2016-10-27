from train import train_linear

from parseData import parseData
from util import getTrainValidSets, normalize, normalizePara
import numpy as np
import sys

# print ('Reading training data')
inFile = sys.argv[1]
if inFile[-3:] != 'csv' :
    inFile = inFile + '.csv'
[feature, target] = parseData(inFile)

# Get training and validation sets
(train_fea, test_fea) = getTrainValidSets(feature, 2)
(train_tar, test_tar) = getTrainValidSets(target, 2)

train_fea = feature
train_tar = target

# Feature normalization
(train_fea, m, s) = normalize(train_fea, axis=0)
print(m.shape)
test_fea = normalizePara(test_fea, m, s)
# print ('Training')
# Training
beta = train_linear(train_fea, train_tar, test_fea, test_tar, eta=1e-5, lamb=0, maxIter=1000000, debug=0)
# Write beta
modFile = open(sys.argv[2], 'w')
beta = beta.tolist()
m = m.tolist()
s = s.tolist()
modFile.write(','.join(str(b[0]) for b in beta) + '\n')
modFile.write(','.join(str(a) for a in m) + '\n')
modFile.write(','.join(str(a) for a in s) + '\n')
