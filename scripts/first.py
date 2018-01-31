import os
import pickle
import random

import gym
import tensorflow as tf

from model.gym_ext import GymWrapper, choose_random_action, ObservationLastStateEncoder
from model.tf_model import TFLearnSpec
from pcams.content_addressable_memory import ContentAddressableMemory


def get_samples(env, num_samples):
    wrapper = GymWrapper(env, num_samples, -10.)
    while not wrapper.at_capacity:
        wrapper.step(choose_random_action(env))

    return list(wrapper.queue)


def cache_get(function, path_builder_func, *args, **kwargs):
    path = path_builder_func(*args, **kwargs)
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
    except (IOError, EOFError):
        data = function(*args, **kwargs)
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, mode='wb') as f:
            pickle.dump(data, f)

    return data


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    data = cache_get(get_samples, lambda **args: 'data/{}.pkl'.format(args), num_samples=5000, env=env)

    print("count ", len(data), "left ", len([x for x in data if x.last_action[1] > x.last_action[0]]), "terminal ",
          len([x for x in data if x.terminal[0] > 0.5]))

    encoder = ObservationLastStateEncoder(data[0])
    cam = ContentAddressableMemory(encoder)
    cam.add_batch(data)

    train = data[:len(data) / 2]
    test = data[len(train):]

    session = tf.Session()
    spec = TFLearnSpec(env, session, "models/first.ckpt")

    spec.train_till_convergence(train, test, max_continues=10)