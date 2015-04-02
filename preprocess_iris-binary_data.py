#!/usr/bin/env python
import csv
import numpy
import cPickle
from scipy import stats

# load data from csv
rawdata = [row for row in csv.reader(open('iris-binary.data', 'rb')) if len(row) > 0]

# partition into thirds
inxs = numpy.random.permutation(len(rawdata))
rawdata = [rawdata[i] for i in inxs]
maxs = []
mins = []
for i in range(len(rawdata)):
	for j in range(len(rawdata[0])-1):
		rawdata[i][j] = float(rawdata[i][j])
		if i == 0:
			maxs.append(rawdata[i][j])
			mins.append(rawdata[i][j])
		else:
			maxs[j] = max(maxs[j],rawdata[i][j])
			mins[j] = min(mins[j],rawdata[i][j])
for i in range(len(rawdata)):
	for j in range(len(rawdata[0])-1):
		rawdata[i][j] = (rawdata[i][j]-mins[j])/maxs[j]

rtrain = rawdata[:50]
rvalid = rawdata[51:75]
rtest = rawdata[76:]

# split X and Y
def split_xy(data):
    x, y = [], []
    for dat in data:
        x.append(dat[:-1])
        y.append(dat[-1])
    return x, y
rtrain_x, rtrain_y = split_xy(rtrain)
rvalid_x, rvalid_y = split_xy(rvalid)
rtest_x, rtest_y = split_xy(rtest)

# transform Xs into numpy arrays
def transform_x(x):
    return stats.zscore(numpy.asarray(x, dtype=numpy.float32))
rtrain_x = transform_x(rtrain_x)
rvalid_x = transform_x(rvalid_x)
rtest_x = transform_x(rtest_x)

# transform (str)Y into (int)Y
def transform_y(iny):
    outy = numpy.zeros(len(iny), dtype=numpy.int16)
    for i in xrange(len(iny)):
        outy[i] = [
            'Iris-setosa',
            'Iris-versicolor'
        ].index(iny[i])
    return outy
rtrain_y = transform_y(rtrain_y)
rvalid_y = transform_y(rvalid_y)
rtest_y = transform_y(rtest_y)

with open('iris-binary.pkl', 'wb') as out_file:
    out_file.write(cPickle.dumps((
        (rtrain_x, rtrain_y),
        (rvalid_x, rvalid_y),
        (rtest_x, rtest_y),
    )))

def try_load():
    with open('iris-binary.pkl', 'rb') as in_file:
        return cPickle.load(in_file)
