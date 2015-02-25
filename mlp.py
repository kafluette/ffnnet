"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import cPickle
import subprocess

import numpy
import theano
import theano.compile
import theano.tensor as T
from scipy.special import gamma
from scipy.linalg import hadamard



# region load_data
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    # ############
    # LOAD DATA #
    #############

    if (not os.path.isfile(dataset)) and dataset == 'iris.pkl':
        import urllib

        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, 'iris.data')
        print 'Running data preprocessing script ...'
        subprocess.call(['python', 'preprocess_iris_data.py'])

    print '... loading data'

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
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
                                               dtype=numpy.int32),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that).
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


# endregion

#region HiddenLayer
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, d, prevHiddenLayer=None, W=None, b=None, S=None, G=None, B=None,
                 activation=T.nnet.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.d = d
        self.n_out = n_out
        self.prevHiddenLayer = prevHiddenLayer

        layer_no = 1
        prev = prevHiddenLayer
        while prev is not None:
            layer_no += 1
            prev = prev.prevHiddenLayer

        dbg_name = lambda s: s + '_l' + str(layer_no)

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        # activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=dbg_name('W'), borrow=True)

        if G is None:
            diag_values = numpy.asarray(rng.normal(0, 1, size=d))
            G_values = numpy.zeros((d, d))
            for i in xrange(d):
                G_values[i, i] = diag_values[i]
            G = theano.shared(value=G_values, name=dbg_name('G'), borrow=True)
        if B is None:
            diag_values = rng.randint(0, 2, d)
            B_values = numpy.zeros((d, d))
            for i in xrange(d):
                B_values[i, i] = -1 if diag_values[i] == 0 else 1
            B = theano.shared(value=B_values, name=dbg_name('B'), borrow=True)
        if S is None:
            S_values = numpy.zeros((d, d))
            for i in xrange(d):
                print "d =", d, ", gamma((d/2)+1) =", gamma((d / 2) + 1)
                s_i = ((2 * numpy.pi) ** (-d / 2)) * (
                    1 / ((numpy.pi ** (d / 2)) / gamma((d / float(2)) + 1)))
                S_values[i, i] = s_i * (numpy.linalg.norm(G_values, ord='fro') ** (-1 / 2))
            S = theano.shared(value=S_values, name=dbg_name('S'), borrow=True)

        self.W = W
        self.S = S
        self.B = B
        self.G = G

        H_values = hadamard(d, dtype=numpy.int)
        H = theano.shared(value=H_values, name='H', borrow=True)
        self.H = H

        perm_matrix_values = numpy.identity(d)  # generated by shuffling the columns of the dxd identity matrix
        numpy.random.shuffle(numpy.transpose(perm_matrix_values))
        perm_matrix = theano.shared(value=perm_matrix_values, name='perm_matrix', borrow=True)
        self.perm_matrix = perm_matrix

        # calculate output
        sigma = 1.0
        d = self.d
        m = self.n_out
        PI = perm_matrix
        phi_inner = 1j * ((1 / sigma * T.sqrt(d)) g* S * H * G * PI * H * B)
        phi_inner *= T.transpose(self.prevHiddenLayer.output if self.prevHiddenLayer is not None else self.input)
        phi = 1 / T.sqrt(m) * T.exp(phi_inner)
        res = activation(T.dot(self.W, T.imag(phi)))
        self.output = res

        self.y_pred = T.argmax(self.output, axis=1)

        # parameters of the model
        self.params = [self.W, self.S, self.B, self.G]

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def cross_entropy_cost(self, y):
        result, updates = theano.scan(
            fn=lambda n: y[n] * T.log(self.output) + (1 - y[n]) * T.log(1 - self.output),
            sequences=[T.arange(y.shape[0])])
        return -T.sum(result)


#endregion

#region MLP
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, d):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            d=d,
            activation=T.nnet.nnet.sigmoid
        )

        self.outputLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_in,
            n_out=n_hidden,
            d=d,
            prevHiddenLayer=None,
            activation=T.nnet.nnet.sigmoid
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.cross_entropy_cost = (
            self.outputLayer.cross_entropy_cost
        )
        # same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.outputLayer.params  #self.hiddenLayer.params + self.outputLayer.params
        # end-snippet-3


#endregion

#region test_mlp
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='iris.pkl', batch_size=5, n_hidden=5):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]

    # for ffnnet stuff to work, the input size must be a power of 2
    # so round up the input size to the next highest power of 2 and pad with zeros as needed
    cur_l = numpy.log2(train_set_x.get_value(borrow=True).shape[1])
    next_l = int(numpy.ceil(cur_l))
    d = 2 ** next_l
    pad_size = d - n_hidden
    if pad_size > 0:
        train_set_x.get_value(borrow=True).resize((n_train, d))
        valid_set_x.get_value(borrow=True).resize((n_valid, d))
        test_set_x.get_value(borrow=True).resize((n_test, d))

    # #####################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.fvector('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)  # TODO

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=train_set_x,
        n_in=4,
        n_hidden=n_hidden,
        n_out=3,
        d=d
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.cross_entropy_cost(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x,
            y: test_set_y
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x,
            y: valid_set_y
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = patience / 2
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for idx in xrange(n_train):
            avg_cost = train_model(idx)

            # iteration number
            iter = (epoch - 1) + idx

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, idx %i/%i, validation error %f %%' %
                    (
                        epoch,
                        idx + 1,
                        n_train,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, idx %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, idx + 1, n_train,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
#endregion
