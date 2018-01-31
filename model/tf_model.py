import os

import tensorflow as tf

from model.common import train_till_convergence


class TFLearnSpec(object):
    def __init__(self, env, session, save_path, load_if_available=True):
        """

        Args:
            session (tf.Session):
            env (gym.make()): AI gym environment
        """
        self._save_path = save_path
        self._session = session
        self._env = env
        self._obs_input_placeholder = tf.placeholder("float", [None] + list(self.observation_tuple))
        self._obs_output_placeholder = tf.placeholder("float", [None] + list(self.observation_tuple))
        self._action_placeholder = tf.placeholder("float", [None, self.actions])
        self._reward_placeholder = tf.placeholder("float", [None, 1])
        self._terminal_placeholder = tf.placeholder("float", [None, 1])
        self._predict_state_op, self._predict_reward_op, self._predict_terminal_op, self._cost_op, self._train_op = build_model(
            self._obs_input_placeholder, self._obs_output_placeholder, self._action_placeholder, self._reward_placeholder,
            self._terminal_placeholder)
        self._saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())

        if load_if_available and os.path.exists(save_path + ".index"):
            self._saver.restore(self._session, save_path)

    @property
    def observation_tuple(self):
        return self._env.observation_space.shape

    @property
    def actions(self):
        return self._env.action_space.n

    @property
    def session(self):
        return self._session

    @property
    def predict_state_op(self):
        return self._predict_state_op

    @property
    def predict_reward_op(self):
        return self._predict_reward_op

    @property
    def predict_terminal_op(self):
        return self._predict_terminal_op

    @property
    def obs_input_placeholder(self):
        return self._obs_input_placeholder

    @property
    def action_placeholder(self):
        return self._action_placeholder

    def train(self, observations):
        # learn that these actions in these states lead to this reward
        _, cost = self._session.run((self._train_op, self._cost_op), feed_dict={
            self._obs_input_placeholder: [x.last_state for x in observations],
            self._obs_output_placeholder: [x.next_state for x in observations],
            self._action_placeholder: [x.last_action for x in observations],
            self._reward_placeholder: [x.reward for x in observations],
            self._terminal_placeholder: [x.terminal for x in observations]})

        return cost

    def predict(self, current_state, action):
        return predict_next_state(self, current_state, action)

    def loss(self, observations):
        return self._session.run(self._cost_op, feed_dict={
            self._obs_input_placeholder: [x.last_state for x in observations],
            self._obs_output_placeholder: [x.next_state for x in observations],
            self._action_placeholder: [x.last_action for x in observations],
            self._reward_placeholder: [x.reward for x in observations],
            self._terminal_placeholder: [x.terminal for x in observations]})

    def train_till_convergence(self, train, test=None, max_continues=4):
        final_cost = train_till_convergence(train, self.train, test, self.loss, max_continues=max_continues)
        self._saver.save(self._session, self._save_path)
        return final_cost


def build_model(obs_input_placeholder, obs_output_placeholder, action_placeholder, reward_placeholder, terminal_placeholder,
                hidden_nodes=(20, 15, 10), non_linarity=tf.nn.relu, learn_rate=0.0001):
    state_dim = obs_output_placeholder.get_shape().as_list()[1]
    input_dim = state_dim + action_placeholder.get_shape().as_list()[1]
    output_dim = state_dim + 1 + 1

    last_dim = input_dim
    last_layer = tf.concat([obs_input_placeholder, action_placeholder], 1)
    for next_dim in hidden_nodes:
        weights = tf.Variable(tf.truncated_normal([last_dim, next_dim], stddev=0.01))
        bias = tf.Variable(tf.constant(0.001, shape=[next_dim]))

        last_layer = non_linarity(
            tf.matmul(last_layer, weights) + bias)
        last_dim = next_dim

    # skip layer
    input_and_modification_layer = tf.concat([last_layer, obs_input_placeholder], 1)

    weights = tf.Variable(tf.truncated_normal([input_and_modification_layer.get_shape().as_list()[1], output_dim], stddev=0.01))
    bias = tf.Variable(tf.constant(0.001, shape=[output_dim]))

    output_layer = tf.matmul(input_and_modification_layer, weights) + bias
    # TODO consider applying sigmoid before cost
    output_target = tf.concat([obs_output_placeholder, reward_placeholder, terminal_placeholder], 1)

    cost_op = tf.reduce_sum(tf.reduce_mean(tf.square(output_target - output_layer), 1))
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(cost_op)

    predict_state_op = tf.slice(output_layer, [0, 0], [-1, state_dim])
    predict_reward_op = tf.slice(output_layer, [0, state_dim], [-1, 1])
    predict_terminal_op = tf.nn.sigmoid(tf.slice(output_layer, [0, state_dim + 1], [-1, 1]))

    return predict_state_op, predict_reward_op, predict_terminal_op, cost_op, train_op


def predict_next_state(tf_learn_spec, current_state, action):
    """

    Args:
        action (int|np.array):
        current_state (np.array):
        tf_learn_spec (TFLearnSpec):
    """
    if isinstance(action, (int, float)):
        action = tf.one_hot(action, tf_learn_spec.actions)

    next_state, reward, terminal = tf_learn_spec.session.run((tf_learn_spec.predict_state_op, tf_learn_spec.predict_reward_op,
                                                              tf_learn_spec.predict_terminal_op),
                                                             feed_dict={
                                                                 tf_learn_spec.obs_input_placeholder: [current_state],
                                                                 tf_learn_spec.action_placeholder: [action]})

    return next_state[0], reward[0], terminal[0]
