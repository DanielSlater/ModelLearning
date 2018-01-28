import numpy as np

from model.gym_ext import get_similar_last_states, Observation


def test_get_similar_last_states():
    observations = [Observation(np.array([0]), None, None, None, None),
                    Observation(np.array([1]), None, None, None, None),
                    Observation(np.array([2]), None, None, None, None),
                    Observation(np.array([3]), None, None, None, None),
                    Observation(np.array([0]), None, None, None, None)]

    similar = get_similar_last_states(Observation(np.array([-1]), None, None, None, None), observations, n=3)

    assert len(similar) == 3
    assert observations[0] in similar
    assert observations[1] in similar
    assert observations[4] in similar
