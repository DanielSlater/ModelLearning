# note must import tensorflow before gym
import random
import pickle
from collections import defaultdict
from collections import deque, namedtuple

import gym
import numpy as np
import tensorflow as tf

BATCH_SIZE = 100

TERMINAL_REWARD = -1.

env = gym.make('CartPole-v0')

ACTIONS_COUNT = 2
FUTURE_REWARD_DISCOUNT = 0.9
LEARN_RATE_ACTOR = 0.01
STORE_SCORES_LEN = 5
GAMES_PER_TRAINING = 3
OBSERVATION_DIMS = env.observation_space.shape[0]
INPUT_NODES = OBSERVATION_DIMS + ACTIONS_COUNT
OUTPUT_NODES = OBSERVATION_DIMS + 1

HIDDEN_1 = 20
HIDDEN_2 = 20
HIDDEN_3 = 20

session = tf.Session()

obs_input_placeholder = tf.placeholder("float", [None, OBSERVATION_DIMS])
obs_output_placeholder = tf.placeholder("float", [None, OBSERVATION_DIMS])
action_placeholder = tf.placeholder("float", [None, ACTIONS_COUNT])
reward_placeholder = tf.placeholder("float", [None, 1])
terminal_placeholder = tf.placeholder("float", [None, 1])


predict_state_op, predict_reward_op, predict_terminal_op, cost_op, train_op = build_model(
    obs_input_placeholder, obs_output_placeholder, action_placeholder, reward_placeholder, terminal_placeholder)

scores = deque(maxlen=STORE_SCORES_LEN)

# set the first action to do nothing
last_action = np.zeros(ACTIONS_COUNT)
last_action[1] = 1

time = 1

session.run(tf.initialize_all_variables())


def choose_random_action():
    return random.randint(0, ACTIONS_COUNT-1)


def train(last_states, last_actions, next_states, reward, terminal):
    # learn that these actions in these states lead to this reward
    _, cost = session.run((train_op, cost_op), feed_dict={
        obs_input_placeholder: last_states,
        obs_output_placeholder: next_states,
        action_placeholder: last_actions,
        reward_placeholder: reward,
        terminal_placeholder: terminal})

    return cost


def predict_next_state(current_state, action):
    if isinstance(action, (int, float)):
        action = one_hot(action, ACTIONS_COUNT)

    next_state, reward, terminal = session.run((predict_state_op, predict_reward_op, predict_terminal_op),
                                               feed_dict={
                                                   obs_input_placeholder: [current_state],
                                                   action_placeholder: [action]})

    return next_state[0], reward[0], terminal[0]


def monte_carlo_tree_search(start_state, apply_action_func, get_available_actions, number_of_samples, max_path_length):
    action_results = defaultdict(float)

    starting_actions = get_available_actions(start_state)
    for start_action in starting_actions:

        first_state, reward, terminal = apply_action_func(start_state, start_action)

        if reward < -0.8: #terminal:
            action_results[start_action] = reward
        else:
            # do rollout
            for _ in range(number_of_samples / len(starting_actions)):
                terminal = False
                next_reward = 0.
                score = reward
                next_state = first_state
                while next_reward > -0.8 and score < max_path_length: # terminal
                    action = random.sample(get_available_actions(next_state), 1)[0]
                    next_state, next_reward, terminal = apply_action_func(next_state, action)
                    score += next_reward

                action_results[start_action] += score

    return max(action_results.iteritems(), key=lambda x: x[1])[0]


last_state = env.reset()
total_reward = 0

OBSERVATIONS = 100000

observations = deque(maxlen=OBSERVATIONS)
games = 0


def train_one_iteration(observations):
    cost = 0.
    random.shuffle(observations)
    for i in xrange(0, len(observations), BATCH_SIZE):
        batch = observations[i:i + BATCH_SIZE]

        cost += train(list(x.last_state for x in batch),
                      list(x.last_action for x in batch),
                      list(x.next_state for x in batch),
                      list(x.reward for x in batch),
                      list(x.terminal for x in batch))

    return cost


def train_till_convergence(observations, max_continues=2):
    best_cost = 100000000.
    continues = 0
    first = True
    while True:
        cost = train_one_iteration(observations)
        if first:
            print("first cost:", cost / len(observations))
            first = False

        if cost < best_cost:
            best_cost = cost
            continues = 0
        elif continues >= max_continues:
            print("last cost:", cost / len(observations))
            return cost
        else:
            continues += 1


game_len = 0

while True:
    env.render()
    if time <= 1000:
        if time == 1000:
            with open('cart_pole_samples.pkl', mode='wb') as f:
                pickle.dumps(observations)
                exit()
        last_action = choose_random_action()
    else:
        monte_carlo_tree_search(last_state, predict_next_state, lambda x: [0, 1], 100, 10)

    current_state, reward, terminal, info = env.step(last_action)

    if terminal:
        reward = TERMINAL_REWARD

    observations.append(Observation(last_state, one_hot(last_action, ACTIONS_COUNT), current_state, [reward], [float(terminal)]))

    if (time % 100) == 0:
        # train
        copy = list(observations)
        train_till_convergence(copy)

    time += 1

    # update the old values
    if terminal:
        last_state = env.reset()
        print game_len
        game_len = 0
    else:
        game_len += 1
        last_state = current_state
