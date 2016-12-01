from train import TFIDF_train
import numpy as np
import sys
import os
maxIter = 1

checkFile = open(os.path.join(sys.argv[1], "check_index.csv"), "r")
checkData = [l.rstrip().split(",") for l in checkFile]
trainFile = open(os.path.join(sys.argv[1], "title_StackOverflow.txt"), "r")
trainData = [l for l in trainFile]

del checkData[0]
print("shape of testData: %d" % len(checkData))

# predictions = []
for ct1 in range(maxIter):
    print("Iter: %d" % ct1)
    predictions = TFIDF_train(trainData, checkData)

# Save results for later testings
# np.save("predictions", predictions)
# predictions = [sum(x)/float(maxIter) for x in zip(*predictions)]

with open(sys.argv[2], "w") as submit:
    submit.write(",".join(["ID", "Ans"])+"\n")
    ct1 = 0
    for l in predictions:
        # tmp = int(predictions[ct1] > 0.5)
        tmp = predictions[ct1]
        submit.write(",".join([str(ct1), str(tmp)])+"\n")
        ct1 += 1

checkFile.close()
trainFile.close()
