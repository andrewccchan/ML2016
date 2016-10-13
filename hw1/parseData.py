
import csv
import numpy as np
import string

def parseData(fileName) :
    # Parameter settings
    zeroIdx = 3
    trainLen = 9
    factors = ['PM2.5']

    # Parse data
    trainData = []
    gndData = []
    with open(fileName, 'r', encoding="big5") as fReader :
        f = csv.reader(fReader, delimiter='\n', quotechar='|')
        for line in f :
            fields = line[0].split(',')
            if fields[2] not in factors :
                continue
            data = fields[zeroIdx : zeroIdx + trainLen]
            trainData.append(list(map(float, data)))
            gndData.append(float(fields[zeroIdx + trainLen]))

    trainData = np.asarray(trainData)
    gndData = np.asarray(gndData)

    return [trainData, gndData]

def getSlidingData(fields, startIdx, endIdx, length) :
    trainData = []
    gndData = []
    for i in range(startIdx, endIdx+1) :
        data = fields[i : i + length]
        trainData.append(list(map(float, data)))
        gndData.append(float(fields[i + length]))

    trainData = np.asarray(trainData)
    gndData = np.asarray(gndData)

    # pwtrainData = np.power(trainData, 2)
    # trainData = np.concatenate((trainData, pwtrainData), axis=1)
    return (trainData, gndData)

def parseData_sliding(fileName) :
    # Parameter settings
    dateIdx = 0
    partIdx = 2
    zeroIdx = 0
    trainLen = 9
    endIdx = 14 # start index of the last 10 hours
    factors = ['PM2.5', 'PM10', 'AMB_TEMP', 'NO2']
    # factors = ['PM2.5', 'NO2', 'PM10']
    # factors = ['PM2.5']

    # factors = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']

    # factors = ['PM2.5']


    # Parse data
    data = dict()
    for i in factors :
        data.setdefault(i, [])

    trainData = 0
    gndData = np.empty((0, 1))
    with open(fileName, 'r', encoding="big5") as fReader :
        f = csv.reader(fReader, delimiter='\n', quotechar='|')
        curDate = ''
        tmp = np.empty((endIdx-zeroIdx+1, 0))
        tmpGND = np.empty((endIdx-zeroIdx+1, 0))
        next(f)
        for line in f :
            fields = line[0].split(',')
            if fields[2] in factors :
                data[fields[2]].extend(fields[3:])

    idxorder = []
    # Perpare data
    for key, val in data.items() :
        idxorder.append(key)
        if key == 'PM2.5' :
            (ret1, ret2) = getSlidingData(val, 0, len(val)-trainLen-1, trainLen)
            trainData = ret1 if type(trainData).__module__ != np.__name__ else np.concatenate((trainData, ret1), axis=1)
            gndData = np.expand_dims(ret2, 1)
        else :
            (ret1, ret2) = getSlidingData(val, 0, len(val)-trainLen-1, trainLen)
            trainData = ret1 if type(trainData).__module__ != np.__name__ else np.concatenate((trainData, ret1), axis=1)
    # for i in range(trainData.shape[0]) :
    #     print(trainData[i,:])
    #
    # for ct1 in range(trainData.shape[0]) :
    #     print(trainData[ct1,:])
    # trainData = np.concatenate((dum, trainData), axis=1)
    return [trainData, gndData, idxorder, trainLen]
