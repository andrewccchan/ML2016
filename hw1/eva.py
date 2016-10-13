import csv
import numpy as np
from util import normalize

def evaluate(fileName, beta, factors, trainLen) :
    zeroIdx = 2
    # trainLen = 9
    # factors = ['PM2.5', 'AMB_TEMP', 'PM10', 'O3', 'SO2']
    # factors = ['PM2.5', 'PM10', 'AMB_TEMP', 'NO2']

    # factors = ['PM2.5']

    # factors = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    # Parse data
    data = dict()
    for i in factors :
        data.setdefault(i, [])

    testData = 0
    init = False
    with open(fileName, 'r', encoding="big5") as fReader :
        f = csv.reader(fReader, delimiter='\n', quotechar='|')
        curID = ''
        tmp = np.empty((1, 0))
        for line in f :
            fields = line[0].split(',')
            if fields[0] != curID :
                curID = fields[0]
                if init :
                    tmp = []
                    for f in factors :
                        tmp.extend(data[f])
                    tmp = np.asarray(tmp)
                    tmp = np.expand_dims(tmp, 1)
                    tmp = tmp.T
                    testData = tmp if type(testData).__module__ != np.__name__ else np.concatenate((testData, tmp), axis=0)
            if fields[1]  in factors :
                tmpdata = fields[-1*trainLen:];
                data[fields[1]] = list(map(float, tmpdata))
            init = True
        # Last row
        tmp = []
        for f in factors :
            tmp.extend(data[f])
        tmp = np.asarray(tmp)
        tmp = np.expand_dims(tmp, 1)
        tmp = tmp.T
        testData = tmp if type(testData).__module__ != np.__name__ else np.concatenate((testData, tmp), axis=0)

    # f = open('eva_debug.txt', 'w')
    # for i in range(testData.shape[0]) :
    #     wd = testData[i,:].tolist()
    #     wd = list(map(str, wd))
    #     f.write(','.join(wd) + '\n')
    #
    # f.close()
    # testData = np.concatenate((dum, testData), axis=1)
    # normalize testData
    gndHat = np.dot(testData, beta)

    submit = open('submit.csv', 'w')
    header = ['id', 'value']
    submit.write(','.join(header) + '\n')

    for ct1 in range(gndHat.shape[0]) :
        data = float(np.asscalar(gndHat[ct1]))
        # if data < 0 :
        #     data = 0
        cont = ['id_' + str(ct1), str(data)]
        submit.write(','.join(cont) + '\n')
