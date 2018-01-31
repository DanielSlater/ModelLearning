import random

import numpy as np


def one_hot(index, size):
    array = np.zeros(size)
    array[index] = 1.
    return array


def train_one_iteration(observations, train_func, batch_size, shuffle=True):
    cost = 0.
    if shuffle:
        random.shuffle(observations)
    for i in xrange(0, len(observations), batch_size):
        batch = observations[i:i + batch_size]

        cost += train_func(batch)

    return cost


def train_till_convergence(train, train_func, test=None, eval_func=None, max_continues=2, max_epochs=3000, batch_size=100):
    best_cost = float("inf")
    continues = 0

    if test:
        assert eval_func

    for epochs in range(max_epochs):
        train_cost = train_one_iteration(train, train_func, batch_size=batch_size)

        if test:
            test_cost = train_one_iteration(train, eval_func, batch_size=batch_size, shuffle=False)
            print("Epoch:", epochs, " train cost:", train_cost, " test cost:", test_cost)
        else:
            test_cost = train_cost
            print("Epoch:", epochs, " train cost:", train_cost)

        if test_cost < best_cost:
            best_cost = test_cost
            continues = 0
        elif continues >= max_continues:
            break
        else:
            continues += 1

    return train_cost
