#!/usr/bin/env python
"""
uses the same mnist data set as the mnist_example but uses the algorithms in the ffnnet paper rather than the standard
ones used in other examples

Reference papers:
Learning Fastfood Feature Transforms for Scalable Neural Networks. Bowick-Marchetti, Micol, et. al.
"""

import logging
import time
import sys
import cPickle
import os
import gzip

import numpy
import numpy.random

from ffnnet import *




# set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # ############
    # LOAD DATA #
    # ############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib

        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def test_ffnnet_mnist(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
                      dataset='mnist.pkl.gz', batch_size=20, n_nodes=10, n_layers=5):
    logger.info("Loading dataset %s", dataset)
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # for ffnnet stuff to work, the input size must be a power of 2
    # so round up the input size to the next highest power of 2 and pad with zeros as needed
    cur_l = numpy.log2(train_set_x.shape[1])
    next_l = int(numpy.ceil(cur_l))
    d = 2 ** next_l
    pad_size = d - n_nodes
    if pad_size > 0:
        train_set_x.resize((train_set_x.shape[0], d))
        valid_set_x.resize((train_set_x.shape[0], d))
        test_set_x.resize((train_set_x.shape[0], d))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    logger.info("Building model ...")

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data presented as rasterized images
    y = T.ivector('y')  # labels presented as 1D vector of [int] labels

    # initialize rng
    rng = numpy.random.RandomState()

    # construct MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_layers=n_layers, n_nodes=n_nodes, n_out=10, d=d)

    # the cost to minimize during training, expressed here symbolically
    # cross entropy error function
    cost = (
        classifier.cross_entropy_cost(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiles a Theano function that compiles the errors made on testing a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in the list
    gparams = [T.grad(cost, param) for param in classifier.params]

    # given A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4]
    # C = zip(A, B) = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # compiles a theano function that returns the cost and updates the parameter of the model
    # based on the rules defined in updates
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    logger.info("Training model ...")

    # early stopping params
    patience = 10000  # look at this many examples no matter what
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through validation_frequency count minibatches before checking the network on the validation set (in this case we check every epoch)
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.0
    start_time = time.clock()

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and not done_looping:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index  # iteration number

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print 'epoch {}, minibatch {}/{}, validation error {}'.format(
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.0
                )

                # if we have a new best validation score
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)


if __name__ == '__main__':
    test_ffnnet_mnist()
