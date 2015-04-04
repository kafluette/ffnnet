import pandas
from operator import mul
from functools import partial


def load_data(n_hidden, *args, **kwargs):
    # load positive data and resize to add label
    pos_train = pandas.read_csv('positive_train.csv', header=True)
    neg_train = pandas.read_csv('negative_train.csv', header=True)

    # count rows and dims
    n_pos, d = pos_train.shape
    n_neg = neg_train.shape[0]
    n = n_pos + n_neg

    # add labels, carefully ...

    # merge positive/negative data
    all_data = pos_train.append(neg_train)

    # partition train, test, validate sets
    train_frac, test_frac, valid_frac = 0.7, 0.2, 0.1
    train_offset, test_offset, valid_offset = map(int,
                                                  map(partial(mul, n),
                                                      (train_frac, test_frac,
                                                       valid_frac)))
    train_set, test_set, valid_set = (all_data[0:train_offset],
                                      all_data[train_offset+1:
                                               train_offset+1+test_offset],
                                      all_data[train_offset+1+test_offset+1:
                                               train_offset+1+test_offset+1+valid_offset])

    return train_set, valid_set, test_set


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
