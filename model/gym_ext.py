import pickle
import random
from collections import deque
from collections import namedtuple

from model.common import one_hot

Observation = namedtuple('Observation', ['last_state', 'last_action', 'next_state', 'reward', 'terminal'])


class GymWrapper(object):
    def __init__(self, env, max_steps_to_store, terminal_reward):
        self._env = env
        self._queue = deque(maxlen=max_steps_to_store)
        self._last_state = env.reset()
        self._time = 0
        self._episodes = 0
        self._terminal_reward = terminal_reward

    def step(self, action):
        current_state, reward, terminal, info = self._env.step(action)
        self._time += 1

        if terminal:
            reward = self._terminal_reward

        self._queue.append(
            Observation(self._last_state, one_hot(action, self._env.action_space.n), current_state, [reward], [float(terminal)]))

        if terminal:
            self._last_state = self._env.reset()
            self._episodes += 1
        else:
            self._last_state = current_state

    @property
    def at_capacity(self):
        return self._time >= self._queue.maxlen

    @property
    def queue(self):
        return self._queue

    def as_serializable(self):
        return pickle.dumps(self._queue)


def get_similar_last_states(observation, observations, comparison=lambda a, b: ((a - b) ** 2).sum(), n=1):
    """

    Args:
        observation (Observation):
        observations ([Observation]):
        comparison:
        n (int):

    Returns:
        [Observation] :
    """
    last_state = observation.last_state
    similar = []

    def inner(new_similarity, other_ob):
        for i, (similarity, item) in enumerate(similar):
            if new_similarity < similarity:
                similar.insert(i, (new_similarity, other_ob))
                return

        similar.append((new_similarity, other_ob))

    threshold_diff = 9999999
    for other_ob in observations:
        if observation is other_ob:
            continue

        new_similarity = comparison(last_state, other_ob.last_state)
        if len(similar) <= n or new_similarity < threshold_diff:
            inner(new_similarity, other_ob)
            similar = similar[:n]
            threshold_diff, _ = similar[0]

    return [ob for _, ob in similar]


def choose_random_action(env):
    return random.randint(0, env.action_space.n - 1)
