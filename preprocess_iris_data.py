#!/usr/bin/env python
import csv
import numpy
import cPickle

# load data from csv
rawdata = [row for row in csv.reader(open('iris.data', 'rb')) if len(row) > 0]

# partition into thirds
rtrain = rawdata[:50]
rvalid = rawdata[51:100]
rtest = rawdata[101:]

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
def transform_x(x):  # TODO: scale x?
    return numpy.asarray(x, dtype=numpy.float32)
rtrain_x = transform_x(rtrain_x)
rvalid_x = transform_x(rvalid_x)
rtest_x = transform_x(rtest_x)

# transform (str)Y into (int)Y
def transform_y(iny):
    outy = numpy.zeros(len(iny), dtype=numpy.int16)
    for i in xrange(len(iny)):
        outy[i] = [
            'Iris-setosa',
            'Iris-versicolor',
            'Iris-virginica'
        ].index(iny[i])
    return outy
rtrain_y = transform_y(rtrain_y)
rvalid_y = transform_y(rvalid_y)
rtest_y = transform_y(rtest_y)

with open('iris.pkl', 'wb') as out_file:
    out_file.write(cPickle.dumps((
        (rtrain_x, rtrain_y),
        (rvalid_x, rvalid_y),
        (rtest_x, rtest_y),
    )))

def try_load():
    with open('iris.pkl', 'rb') as in_file:
        return cPickle.load(in_file)
