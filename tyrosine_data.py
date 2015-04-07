import pandas
from operator import mul
from functools import partial
import theano.tensor as T
import theano
import numpy
import sys

def load_data(dataset, n_hidden):
    all_data = pandas.read_csv('tyrosine_train.csv', header=True).as_matrix()
    n, d = all_data.shape
    d -= 1  # don't count y
    print >>sys.stderr, "Loaded tyrosine data, n = %d, d = %d" % (n, d)
    
    all_data_x, all_data_y = all_data[:, range(0, d-1)], all_data[:, [d]]

    # partition train, test, validate sets
    train_frac, valid_frac, test_frac = 0.7, 0.15, 0.15
    train_offset, valid_offset, test_offset = map(int,
                                                  map(partial(mul, n),
                                                      [train_frac, valid_frac,
                                                       test_frac]))
    print >>sys.stderr, "Using split: train %d %%, valid %d %%, test %d %%" % (train_frac * 100.,
                                                                               valid_frac * 100.,
                                                                               test_frac * 100.)
    print >>sys.stderr, "Resulting in # samples: %d train, %d valid, %d test" % (train_offset, valid_offset, test_offset)
    
    train_set, valid_set, test_set = (all_data[0:train_offset],
                                      all_data[train_offset+1:
                                               train_offset+1+valid_offset],
                                      all_data[train_offset+1+valid_offset+1:
                                               train_offset+1+valid_offset+1+test_offset])

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy[:, range(0, d-1)], data_xy[:, d]
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    return shared_dataset(train_set), shared_dataset(valid_set), shared_dataset(test_set)


def test_load_data():
    print "Loading data set ..."
    train, valid, test = load_data()
    print "Checking size ..."
    assert train.shape == (1544, 189)
    assert valid.shape == (220, 189)
    assert test.shape == (441, 189)
    print "Done."


if __name__ == '__main__':
    test_load_data()
