import numpy as np


def write_submit_file(l, fname):
    if len(l) != 606779:
        return
    with open(fname, 'w') as fw:
        fw.write('id,label\n')
        fw.write('\n'.join(['%d,%d' % (i+1, l[i]) for i in range(606779)]))


def get_type_dict(fname='data/training_attack_types.txt', label=False, all_class=False):
    d = {'normal': 0}

    for i, line in enumerate(open(fname)):
        l = line.strip().split()
        attack, type_ = l[0], l[1]

        if all_class:
            d[attack] = i+1
        else:
            if not attack in d:
                d[attack] = []
            d[attack].append(type_)

    if not all_class:
        d['normal'] = [0]
        for key in d:
            d[key] = list(set(d[key]))

    if label and not all_class:
        label_d = {'dos': 1, 'u2r': 2, 'r2l': 3, 'probe': 4}
        for key in d:
            d[key] = [label_d[t] for t in d[key] if t in label_d]
        d['normal'] = [0]

    return d


def reverse_dict(d):
    rev_d = {}
    for key in d:
        rev_d[d[key]] = key

    return rev_d


def get_train_data(fname='data/train.bin.label'):
    X, y = [], []
    for line in open(fname):
        l = line.strip().split(',')
        X.append([float(i) for i in l[:-1]])
        y.append(int(l[-1]))

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def get_test_data(fname='data/test.bin'):
    X = []
    for line in open(fname):
        l = line.strip().split(',')
        X.append([float(i) for i in l])

    X = np.asarray(X)

    return X
