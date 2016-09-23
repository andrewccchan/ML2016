#!/bin/bash

import sys

if len(sys.argv) != 3 :
    raise Exception('There should be two arguments. Format ./Q1.sh [column] [file]')

col = int(sys.argv[1])
fileName = sys.argv[2]
ret = []
ret_str = []
with open(fileName) as f :
    for line in f :
        line = line.strip(' ')
        fields = line.split(' ')
        # check that the specified column exists
        if col >= len(fields) :
            errMsg = 'column ' + str(col) + ' does not exist in ' + line
            raise Exception(errMsg)
        ret.append([float(fields[col]), fields[col]])

ret = sorted(ret, key=lambda k:k[0])

with open('ans1.txt', 'w') as outFile :
    outFile.write(','.join(s[1] for s in ret) + '\n')

