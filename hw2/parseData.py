import itertools
import csv
import numpy as np
import string

def parseData(fileName) :
    feature = 0
    target = 0
    with open(fileName, 'r') as fReader :
        # Get two independent readers
        r1, r2 = itertools.tee(csv.reader(fReader, delimiter=',', quotechar='|'))

        # Get column number, init np array
        nCol = len(next(r2))
        feature = np.empty((0, nCol-2))
        target = np.empty((0, 1))

        # Read file
        for fields in r1 :
            tmp = np.asarray(list(map(float, fields[1:nCol-1])))
            tmp = np.expand_dims(tmp, axis=1)
            feature = np.concatenate((feature, tmp.T), axis=0)
            tmp_tar = np.asarray(list(map(float, fields[nCol-1:nCol])))
            tmp_tar = np.expand_dims(tmp_tar, axis=1)
            target = np.concatenate((target, tmp_tar.T), axis = 0)

        # print (feature)
        return [feature, target]


def parseTestData(fileName) :
    feature = 0
    target = 0
    with open(fileName, 'r') as fReader :
        # Get two independent readers
        r1, r2 = itertools.tee(csv.reader(fReader, delimiter=',', quotechar='|'))

        # Get column number, init np array
        nCol = len(next(r2))
        feature = np.empty((0, nCol-1))

        # Read file
        for fields in r1 :
            tmp = np.asarray(list(map(float, fields[1:nCol])))
            tmp = np.expand_dims(tmp, axis=1)
            feature = np.concatenate((feature, tmp.T), axis=0)

        return feature
