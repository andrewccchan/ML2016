import numpy as np

def getTrainValidSets(data, portion) :
    dataNum = data.shape[0]
    bound = np.asscalar(np.floor(dataNum/portion))
    train = data[0:int(bound), :]
    valid = data[int(bound)+1:, :]
    # print(train)
    return (train, valid)

def normalize(data, axis) :
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    return np.divide((data - m), s)
