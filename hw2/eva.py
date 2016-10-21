import numpy as np
from parseData import parseTestData
from util import normalize, calErr, sigmoid, f, padOne, normalizePara

# def evaluate(fileName, beta, outFile) :
#     feature = parseTestData(fileName)
#     feature = padOne(feature, dim=1)
#
#     gndHat = f(feature, beta)
#
#     submit = open(outFile+'.csv', 'w')
#     header = ['id', 'label']
#     submit.write(','.join(header) + '\n')
#
#     for ct1 in range(gndHat.shape[0]) :
#         # data = np.rint(gndHat[ct1])
#         data = int(np.asscalar(gndHat[ct1]))
#         cont = [str(ct1+1), str(data)]
#         submit.write(','.join(cont) + '\n')

def evaluate(fileName, beta, outFile, m, s) :
    feature = parseTestData(fileName)
    feature = normalizePara(feature, m, s)
    feature = padOne(feature, dim=1)

    # feature = normalize(feature, axis=0)
    gndHat = f(feature, beta)

    # for ct1 in range(gndHat.shape[0]) :
    #     data = np.rint(gndHat[ct1])
    #     print(int(np.asscalar(data)))

    submit = open(outFile+'.csv', 'w')
    header = ['id', 'label']
    submit.write(','.join(header) + '\n')

    for ct1 in range(gndHat.shape[0]) :
        data = np.rint(gndHat[ct1])
        data = int(np.asscalar(data))
        cont = [str(ct1+1), str(data)]
        submit.write(','.join(cont) + '\n')
