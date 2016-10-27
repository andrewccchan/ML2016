from train_ann import train_ann
import numpy as np
from parseData import parseData
from util import getTrainValidSets, normalize, normalizePara
from util import calErr
import sys

inFile = sys.argv[1]
if inFile[-3:] != 'csv' :
    inFile = inFile + '.csv'
[feature, target] = parseData(inFile)

# Get training and validation sets
(train_fea, test_fea) = getTrainValidSets(feature, 1.1)
(train_tar, test_tar) = getTrainValidSets(target, 1.1)

train_fea = feature
train_tar = target

# Feature normalization
(train_fea, m, s) = normalize(train_fea, axis=0)
test_fea = normalizePara(test_fea, m, s)


# Validation
para = train_ann(train_fea.T, train_tar, test_fea.T, test_tar)

# Write parameters
modFile = open(sys.argv[2], 'w')
for p in para :
    modFile.write(str(p.shape[0])+ ','+ str(p.shape[1])+ ',')
    p = np.reshape(p, (1, p.shape[0]*p.shape[1]))
    p = p[0].tolist()
    modFile.write(','.join(str(b) for b in p) + '\n')

m = m.tolist()
s = s.tolist()
modFile.write(str(len(m))+ ','+ '1'+','+','.join(str(a) for a in m) + '\n')
modFile.write(str(len(s))+ ','+ '1'+','+','.join(str(a) for a in s) + '\n')
