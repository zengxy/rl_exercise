import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, min_act: float, max_act: float, feature_size: int,
                 sess: tf.Session, hidden_units, learning_rate):
        self.min_act = min_act
        self.max_act = max_act
        self.sess = sess
        self.feature_size = feature_size
        self.hidden_units = [hidden_units] if isinstance(hidden_units, int) else hidden_units
        self.lr = learning_rate
        self.step_counter = 0
        # self.gamma = gamma
        self._build_network()

    def _build_network(self):
        w_initializer = tf.random_normal_initializer(0., .1)
        b_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope("Actor"):
            self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
            self.td_error_pl = tf.placeholder(tf.float32, shape=[None], name="td_error")
            self.a_pl = tf.placeholder(tf.float32, shape=[None], name="a")
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden",
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer)
            else:
                out = self.s_pl
                for (i, hidden_unit) in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden_unit, activation=tf.nn.relu,
                                          name="hidden_{:d}".format(i),
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

            with tf.variable_scope("out"):
                self.mu = tf.squeeze(tf.layers.dense(out, 1, activation=None,
                                                     kernel_initializer=w_initializer,
                                                     bias_initializer=b_initializer)*2,
                                     name="mu")
                self.sigma = tf.squeeze(tf.layers.dense(out, 1, activation=tf.nn.softplus,
                                                        kernel_initializer=w_initializer,
                                                        bias_initializer=b_initializer)+0.1,
                                        name="sigma")
                self.dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
                self.action = tf.clip_by_value(self.dist.sample(1), self.min_act, self.max_act)

            with tf.variable_scope("loss"):
                log_prob = self.dist.log_prob(self.action)  # loss without advantage
                self.loss = tf.reduce_mean(-log_prob*self.td_error_pl-0.01*self.dist.entropy())
                # advantage (TD_error) guided loss
                # Add cross entropy cost to encourage exploration
                tf.summary.scalar("loss", self.loss, collections=["Actor"])
                self.summary = tf.summary.merge_all(key="Actor")

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, a, td_error, tf_log_writer=None):
        _, summary_val = self.sess.run(
            [self.train_op, self.summary],
            feed_dict={self.s_pl: s, self.a_pl: a, self.td_error_pl: td_error})
        if tf_log_writer:
            tf_log_writer.add_summary(summary_val, self.step_counter)
        self.step_counter += 1

    def choose_action(self, s: np.ndarray):
        if len(s.shape) == 1:
            s = s[np.newaxis, :]
        action_val = self.sess.run(self.action, feed_dict={self.s_pl: s})
        return action_val


class Critic:
    def __init__(self, feature_size: int, sess: tf.Session,
                 hidden_units, learning_rate, gamma):
        self.sess = sess
        self.feature_size = feature_size
        self.hidden_units = [hidden_units] if isinstance(hidden_units, int) else hidden_units
        self.lr = learning_rate
        self.gamma = gamma
        self._build_network()
        self.step_counter = 0

    def _build_network(self):
        w_initializer = tf.random_normal_initializer(0., .1)
        b_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope("Critic"):
            self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
            self.v_next_pl = tf.placeholder(tf.float32, shape=[None], name="v_")
            self.r_pl = tf.placeholder(tf.float32, shape=[None], name="r")
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden",
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer)
            else:
                out = self.s_pl
                for (i, hidden_unit) in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden_unit,
                                          activation=tf.nn.relu, name="hidden_{:d}".format(i),
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

            self.v = tf.squeeze(tf.layers.dense(out, 1, activation=None,
                                                kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer),
                                name="out")

            with tf.variable_scope("loss"):
                self.td_error = self.r_pl + self.gamma * self.v_next_pl - self.v
                self.loss = tf.reduce_mean(tf.square(self.td_error))
                tf.summary.scalar("loss", self.loss, collections=["Critic"])
                tf.summary.scalar("td_error", tf.reduce_mean(self.td_error), collections=["Critic"])
                self.summary = tf.summary.merge_all(key="Critic")

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_, tf_log_writer=None):
        v_ = self.sess.run(self.v, feed_dict={self.s_pl: s_})
        summary_val, _, td_error_val = self.sess.run(
            [self.summary, self.train_op, self.td_error],
            feed_dict={self.s_pl: s, self.v_next_pl: v_.ravel(), self.r_pl: r})
        if tf_log_writer:
            tf_log_writer.add_summary(summary_val, self.step_counter)
        self.step_counter += 1
        return td_error_val
