import pickle
import random

import gym

from model.gym_ext import GymWrapper, choose_random_action


def get_samples(num_samples):
    env = gym.make('CartPole-v0')
    wrapper = GymWrapper(env, num_samples, -10.)
    while not wrapper.at_capacity:
        wrapper.step(choose_random_action(env))

    return wrapper.queue


def cache_get(function, path_builder_func, *args, **kwargs):
    path = path_builder_func(*args, **kwargs)
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
    except (IOError, EOFError):
        data = function(*args, **kwargs)
        with open(path, mode='wb') as f:
            pickle.dump(data, f)

    return data

if __name__ == '__main__':
    data = cache_get(get_samples, lambda **num_samples: 'data/cart_pole_samples_{}.pkl'.format(num_samples), num_samples=5000)

    print("count ", len(data), "left ", len([x for x in data if x.last_action[1] > x.last_action[0]]), "terminal ",
          len([x for x in data if x.terminal[0] > 0.5]))
    random.shuffle(data)

    train = data[:len(data) / 2]
    test = data[len(train):]

    TF