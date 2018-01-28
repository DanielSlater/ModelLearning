import random

import numpy as np


def one_hot(index, size):
    array = np.zeros(size)
    array[index] = 1.
    return array


def train_one_iteration(observations, train_func, batch_size):
    cost = 0.
    random.shuffle(observations)
    for i in xrange(0, len(observations), batch_size):
        batch = observations[i:i + batch_size]

        cost += train_func((x.last_state for x in batch),
                           (x.last_action for x in batch),
                           (x.next_state for x in batch),
                           (x.reward for x in batch),
                           (x.terminal for x in batch))

    return cost


# TODO: test set?
def train_till_convergence(observations, train_func, max_continues=2, max_epochs=1000, batch_size=100):
    best_cost = 100000000.
    continues = 0
    first = True

    for epochs in range(max_epochs):
        cost = train_one_iteration(observations, train_func, batch_size=batch_size)

        if first:
            print("first cost:", cost / len(observations))
            first = False

        if cost < best_cost:
            best_cost = cost
            continues = 0
        elif continues >= max_continues:
            break
        else:
            continues += 1

    print("Epochs ", epochs, " last cost:", cost / len(observations))
    return cost
