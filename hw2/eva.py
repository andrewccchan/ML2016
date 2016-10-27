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
with open(modFile) as mf :
    tmp = mf.readline().rstrip('\n')
    tmp = tmp.split(',')
    beta = np.asarray(list(map(float, tmp)))
    beta = np.reshape(beta, (len(tmp), 1))

    tmp = mf.readline().rstrip('\n')
    tmp = tmp.split(',')
    m = np.asarray(list(map(float, tmp)))

    tmp = mf.readline().rstrip('\n')
    tmp = tmp.split(',')
    s = np.asarray(list(map(float, tmp)))


feature = parseTestData(tstFile)
feature = normalizePara(feature, m, s)
feature = padOne(feature, dim=1)

# feature = normalize(feature, axis=0)
gndHat = f(feature, beta)

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
