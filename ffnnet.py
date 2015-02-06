"""
from:
    http://www.deeplearning.net/tutorial/mlp.html#mlp
"""
__docformat__ = 'restructedtext en'

import itertools

import numpy
import numpy.linalg
import numpy.random
import theano
import theano.tensor as T
from scipy.linalg import hadamard
from scipy.special import gamma

from logistic_sgd import LogisticRegression


class HiddenLayer(object):
    def __init__(self, layer_no, num_layers, rng, input, n_in, n_out, d, W=None, b=None, S=None, G=None, B=None):
        self.input = input

        if W is None:
            # initialize using IWI method
            # (https://www.researchgate.net/publication/262678356_Interval_based_Weight_Initialization_Method_for_Sigmoidal_Feedforward_Artificial_Neural_Networks)
            W_values = numpy.asarray(
                rng.uniform(
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

        if G is None:
            diag_values = numpy.asarray(rng.normal(0, 1, size=d))
            G_values = numpy.zeros((d, d))
            for i in xrange(d):
                G_values[i, i] = diag_values[i]
            G = theano.shared(value=G_values, name='G', borrow=True)
        if B is None:
            diag_values = rng.randint(0, 1, size=d)
            B_values = numpy.zeros((d, d))
            for i in xrange(d):
                B_values[i, i] = diag_values[i]
            B = theano.shared(value=B_values, name='B', borrow=True)
        if S is None:
            S_values = numpy.zeros((d, d))
            for i in xrange(d):
                s_i = ((2 * numpy.pi) ** (-d / 2)) * (1 / ((numpy.pi ** (d / 2)) / gamma((d / 2) + 1)))
                S_values[i, i] = s_i * (numpy.linalg.norm(G.get_value(borrow=True), ord='fro') ** (-1 / 2))

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
    def __init__(self, rng, input, n_in, n_layers, n_nodes, n_out, d, output_clz=LogisticRegression):
        self.input = input
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_in = n_in
        self.n_out = n_out
        self.d = d

        # generate ffnnet parameters perm_matrix (random permutation matrix) and H (hadamard matrix)
        H_values = hadamard(d, dtype=numpy.int)
        H = theano.shared(value=H_values, name='H', borrow=True)
        self.H = H

        perm_matrix_values = numpy.identity(d)  # generated by shuffling the columns of the dxd identity matrix
        numpy.random.shuffle(numpy.transpose(perm_matrix_values))
        perm_matrix = theano.shared(value=perm_matrix_values, name='perm_matrix', borrow=True)
        self.perm_matrix = perm_matrix

        # generate the first hidden layer, taking as inputs the input nodes
        self.hiddenLayers = []
        self.hiddenLayers.append(HiddenLayer(
            layer_no=0 + 1,  # IWI indexing starts at 1
            num_layers=n_layers,
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_nodes,
            d=d
        ))

        # generate the rest of the hidden layers, taking as inputs the previous layer's output
        for l in xrange(1, n_layers):
            self.hiddenLayers.append(HiddenLayer(
                layer_no=l + 1,
                num_layers=n_layers,
                rng=rng,
                input=self.hiddenLayer[l - 1].output,
                n_in=n_in,
                n_out=n_nodes,
                d=d,
            ))

        # The output layer gets as input the units of the last hidden layer
        self.outputLayer = output_clz(
            input=self.hiddenLayers[n_layers - 1].output,
            n_in=n_nodes,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            (abs(self.hiddenLayers[i].W).sum() for i in xrange(n_layers)).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayers.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        # number of errors of the MLP
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the layers it is made out of (hiddens + output)
        self.params = itertools.chain(
            *(self.hiddenLayers[i].params for i in xrange(n_layers))) + self.outputLayer.params

    def cross_entropy_cost(self, y):
        pass
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        #results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
        #compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=[results])
