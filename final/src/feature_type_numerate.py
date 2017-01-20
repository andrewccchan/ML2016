from utility import *

import sys

f = sys.argv[1]
o = sys.argv[2]
all_types = True if len(sys.argv) > 3 else False

d = get_type_dict(label=True, all_class=all_types)

fw = open(o, 'a')

for line in open(f):
    feature = line.strip().strip('.').split(',')
    type_ = feature[-1]

    if not all_types:
        for label in d[type_]:
            feature[-1] = str(label)
            fw.write(','.join(feature) + '\n')

    else:
        feature[-1] = str(d[type_])
        fw.write(','.join(feature) + '\n')

fw.close()
