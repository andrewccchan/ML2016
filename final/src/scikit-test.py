from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utility import *

import numpy as np
import sys
import argparse

def train(args):
    print('training GB for arguments:\n', args)
    
    n = args.n_estimators
    s = args.subsample
    l = args.learning_rate
    d = args.max_depth
    f = args.max_features

    if f and f not in ['auto', 'sqrt', 'log2']:
        if '.' in f:
            f = float(f)
        else:
            f = int(f)

    verbose = 2
    
    print('Getting training data...')
    X, y = get_train_data()

    cls = GradientBoostingClassifier(n_estimators=n,
                                    subsample=s,
                                    learning_rate=l,
                                    max_depth=d,
                                    max_features=f,
                                    verbose=verbose)

    print('Fitting model...')
    cls.fit(X, y)

    model_name = 'models/gb_n{:d}_s{:.2f}_l{:.2f}_d{:d}_f{:s}.pkl'.format(n, s, l, d, str(f))

    print('Saving model: {:s}...'.format(model_name))
    joblib.dump(cls, model_name)


def test(model):
    
    print('Getting training data...')
    X, y = get_train_data()
    
    print('\n\nGetting testing data...')
    tX = get_test_data()
    
    for line in open(model):
        cls = joblib.load('models/' + line.strip())
        
        print('Predicting training data...')
        y_pred = cls.predict(X)
        
        print('\nAccuracy on training data: {:.6f}'.format(np.mean(y==y_pred)))
        print('\nDistribution of predicted label:\n', np.bincount(y_pred)/y_pred.shape[0])
        print('\nConfusion matrix:\n', confusion_matrix(y, y_pred))

        
        print('Predicting testing data...')
        ty_pred = cls.predict(tX)

        print('\nDistribution of predicted labels:\n', np.bincount(ty_pred)/ty_pred.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_estimators', 
                        help='number of estimators', 
                        default=100, 
                        type=int)
    parser.add_argument('-s', '--subsample', 
                        help='subsample fraction', 
                        default=1.0, 
                        type=float)
    parser.add_argument('-l', '--learning_rate', 
                        help='learning rate', 
                        default=0.1, 
                        type=float)
    parser.add_argument('-d', '--max_depth', 
                        help='max depth', 
                        default=3, 
                        type=int)
    parser.add_argument('-f', '--max_features', 
                        help='max features', 
                        default=None, 
                        type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-m', '--model', help='file name contains model names for testing')

    args = parser.parse_args()
    if args.test and args.model:
        test(args.model)
    elif not args.test:
        train(args)
    else:
        print('You must specify both --test and --model')
