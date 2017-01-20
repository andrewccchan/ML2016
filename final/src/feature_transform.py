from sklearn.preprocessing import LabelBinarizer

import numpy as np
import sys

f = sys.argv[1]
discrete_index = [1, 2, 3, 6, 11, 20, 21]

test = True if len(sys.argv) > 2 else False

discrete_features = []

features = [line.strip().split(',') for line in open(f)]

for idx in discrete_index:
    discrete_features.append([line[idx] for line in features])

if test:
    discrete_test = []
    new_test = []
    f_test = [line.strip().split(',') for line in open('data/test.in')]
    for idx in discrete_index:
        discrete_test.append([line[idx] for line in f_test])

new_features = []

for i, idx in enumerate(discrete_index):
    lb = LabelBinarizer()
    new_features.append(lb.fit_transform(discrete_features[i]))
    print(idx, np.unique(discrete_features[i], return_counts=True))

    if test:
        new_test.append(lb.transform(discrete_test[i]))
        print(idx, np.unique(discrete_test[i], return_counts=True))


for i, idx in enumerate(discrete_index):
    if not test:     #for j in range(len(features)):
        for j in range(len(features)):
            features[j][idx] = ','.join([str(n) for n in new_features[i][j]])

    if test:
        for j in range(len(f_test)):
            f_test[j][idx] = ','.join([str(n) for n in new_test[i][j]])

if not test:
    with open(f+'.bin', 'a') as fw:
        for feature in features:
            fw.write(','.join(feature) + '\n')

if test:
    with open('test.bin', 'a') as fw:
        for feature in f_test:
            fw.write(','.join(feature) + '\n')
