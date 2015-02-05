"""
from:
    http://www.deeplearning.net/tutorial/mlp.html#mlp
"""
__docformat__ = 'restructedtext en'

import itertools

import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression


class HiddenLayer(object):
    def __init__(self, layer_no, num_layers, rng, input, n_in, n_out, W=None, b=None, S=None, G=None, B=None):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(  # initialize using IWI method
                              low=(2 * layer_no - 1) / (num_layers - 1),
                              high=(2 * layer_no + 1) / (num_layers - 1),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        if S is None:
            pass
        if G is None:
            pass
        if B is None:
            pass

        self.W = W
        self.b = b

        # FFNNet params
        self.S = S
        self.G = G
        self.B = B

        lin_output = T.dot(input, self.W) + self.b
        self.output = T.nnet.sigmoid(lin_output)

        # parameters of the model
        self.params = [self.W, self.b, self.S, self.G, self.B]


class MLP(object):
    def __init__(self, rng, input, n_in, n_layers, n_nodes, n_out, output_clz=LogisticRegression):
        # generate the first hidden layer, taking as inputs the input nodes
        self.hiddenLayer[0] = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_nodes,
            activation=T.sigmoid
        )

        # generate the rest of the hidden layers, taking as inputs the previous layer's output
        for l in xrange(1, n_layers):
            self.hiddenLayer[l] = HiddenLayer(
                rng=rng,
                input=self.hiddenLayer[l - 1].output,  # TODO do transform here (not above, in HiddenLayer)
                n_in=n_in,
                n_out=n_nodes,
                activation=T.sigmoid
            )

        # The output layer gets as input the hidden units of the hidden layer
        self.outputLayer = output_clz(
            input=self.hiddenLayer[n_layers - 1].output,
            n_in=n_nodes,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        # TODO is this weighting the correct way to do it? (also for L2)
        self.L1 = (
            (abs(self.hiddenLayer[i].W).sum() for i in xrange(n_layers)).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # cross entropy cost of the MLP computed in the output layer
        self.cross_entropy_cost = (
            self.outputLayer.cross_entropy_cost
        )
        # same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the layers it is made out of (hiddens + output)
        self.params = itertools.chain(*(self.hiddenLayer[i].params for i in xrange(n_layers))) + self.outputLayer.params
