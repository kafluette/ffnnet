import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("A")
y = T.ivector("y")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda n: (1-y[n]) * T.log(T.nnet.sigmoid(1)),sequences=[T.arange(10)])

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result.sum()

# compiled function that returns A**k
power = theano.function(inputs=[y], outputs=final_result, updates=updates)

print power(range(10))
print power(range(10))
