import sys

if len(sys.argv) != 3 :
    raise Exception('There should be two arguments. Format ./Q1.sh [column] [file]')
try :
    col = int(sys.argv[1])
except ValueError :
    raise Exception('column index should be an integer')
fileName = sys.argv[2]
ret = []
ret_str = []
with open(fileName) as f :
    for line in f :
        line = line.strip(' ')
        line = line.strip('\n')
        fields = line.split(' ')
        # check that the specified column exists
        if col >= len(fields) or col < 0:
            errMsg = 'column ' + str(col) + ' exceeds boundary ' 
            raise Exception(errMsg)
        ret.append([float(fields[col]), fields[col]])

ret = sorted(ret, key=lambda k:k[0])

with open('ans1.txt', 'w') as outFile :
    outFile.write(','.join(s[1] for s in ret) + '\n')
