import numpy as np
import tensorflow as tf


class PolicyGradient:
    def __init__(self, min_action, max_action, feature_size,
                 hidden_units=10, learning_rate=0.01, gamma=0.95, tf_log_dir=None):
        self.min_action = min_action
        self.max_action = max_action
        self.feature_size = feature_size
        self.hidden_units = [hidden_units] if isinstance(hidden_units, int) else hidden_units
        self.lr = learning_rate
        self.gamma = gamma
        self.step_counter = 0
        self.log_writer = None
        self._build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        if tf_log_dir:
            self.log_writer = tf.summary.FileWriter(tf_log_dir, self.sess.graph)

    def _build_network(self):
        self.s_pl = tf.placeholder(tf.float32, [None, self.feature_size], name="s")
        self.a_pl = tf.placeholder(tf.float32, [None], name="a")
        self.v_pl = tf.placeholder(tf.float32, [None], name="v_discounted")

        if len(self.hidden_units) == 1:
            out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                  activation=tf.nn.relu, name="hidden")
        else:
            out = self.s_pl
            for (i, hidden_unit) in enumerate(self.hidden_units):
                out = tf.layers.dense(out, hidden_unit,
                                      activation=tf.nn.relu, name="hidden_{:d}".format(i))

        with tf.variable_scope("out"):
            self.mu = tf.squeeze(tf.layers.dense(out, units=1), name="mu")
            self.sigma = tf.squeeze(tf.layers.dense(out, units=1, activation=tf.nn.softplus), name="sigma")
            self.dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma+0.001)
            self.action = tf.clip_by_value(
                self.dist.sample(sample_shape=1), self.min_action, self.max_action)

        with tf.variable_scope("loss"):
            self.loss = -self.dist.log_prob(self.action)*self.v_pl - 0.01*self.dist.entropy()

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, s):
        if len(s.shape) == 1:
            s = s[np.newaxis, :]
        return self.sess.run(self.action, feed_dict={self.s_pl: s})

    def store_transition(self, s, a, r):
        self.ep_states.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def learn(self):
        discounted_r = self._discounted_rewards()
        self.sess.run(self.train_op, feed_dict={
            self.s_pl: np.vstack(self.ep_states),  # shape=[None, n_obs]
            self.a_pl: np.array(self.ep_actions).reshape([-1]),  # shape=[None, ]
            self.v_pl: discounted_r,  # shape=[None, ]
        })

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []
        # return discounted_ep_rs_norm

    def _discounted_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
