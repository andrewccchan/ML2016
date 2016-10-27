import numpy as np
from parseData import parseTestData
from util import normalize, calErr, sigmoid, f, padOne, normalizePara
import sys

def procInput(s) :
    if s[-3:] != 'csv' :
        return s +'.csv'
    else :
        return s

modFile = sys.argv[1]
tstFile = procInput(sys.argv[2])
outFile = procInput(sys.argv[3])

# parse model file
para = []
with open(modFile) as mf :
    for l in mf :
        ct = l.split(',')
        tmp = ct[2:]
        tmp = np.asarray(list(map(float, tmp)))
        tmp = np.reshape(tmp, (int(ct[0]), int(ct[1])))
        para.append(tmp)

w1 = para[0]
w2 = para[1]
b1 = para[2]
b2 = para[3]
m = np.squeeze(para[4])
s = np.squeeze(para[5])

feature = parseTestData(tstFile)
feature = normalizePara(feature, m, s).T

gndHat = []
# feature = normalize(feature, axis=0)
for ct1 in range(feature.shape[1]) :
    x = feature[:, ct1:ct1+1]

    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    gndHat.append(a2)

gndHat = np.squeeze(np.asarray(gndHat), axis=2)
# for ct1 in range(gndHat.shape[0]) :
#     data = np.rint(gndHat[ct1])
#     print(int(np.asscalar(data)))

submit = open(outFile, 'w')
header = ['id', 'label']
submit.write(','.join(header) + '\n')

for ct1 in range(gndHat.shape[0]) :
    data = np.rint(gndHat[ct1])
    data = int(np.asscalar(data))
    cont = [str(ct1+1), str(data)]
    submit.write(','.join(cont) + '\n')
