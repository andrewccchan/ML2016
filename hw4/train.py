from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import nltk
import os
from nltk.stem.porter import PorterStemmer
import string

def TFIDF_train(rawInput, testData):
    # Cleaning up data
    print("Cleaning up input...")
    data = []
    for l in rawInput:
        tmp = l.lower().rstrip().translate(None, string.punctuation)
        data.append(tmp)

    # print(data)
    print("Performing feature extraction")
    max_df = 0.5 # in terms of the whole document
    min_df = 2 # in terms of exact work count
    use_idf = True
    tfidf = TfidfVectorizer(max_df=max_df, max_features=None,
                            min_df=min_df, use_idf=use_idf,
                            stop_words='english')
    X = tfidf.fit_transform(data)

    print("Perform dimenation reduction using LSA")
    svd = TruncatedSVD(20) # 20
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)
    print("Dimension reduced to %d" % X.shape[1])

    print("Clustering mini-batch data")
    km = KMeans(n_clusters=25, init='k-means++',
                verbose=False, n_init=100)
    labels = km.fit_predict(X)

    results = []

    for t in testData:
        t = map(int, t)
        # print t
        tmp = 1 if labels[t[1]] == labels[t[2]] else 0
        results.append(tmp)

    return results
