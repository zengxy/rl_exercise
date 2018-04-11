import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, actions: list, feature_size: int,
                 sess: tf.Session, hidden_units, learning_rate):
        self.actions = actions
        self.action_num = len(actions)
        self.sess = sess
        self.feature_size = feature_size
        self.hidden_units = [hidden_units] if isinstance(hidden_units, int) else hidden_units
        self.lr = learning_rate
        self.step_counter = 0
        # self.gamma = gamma
        self._build_network()

    def _build_network(self):
        with tf.variable_scope("Actor"):
            self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
            self.td_error_pl = tf.placeholder(tf.float32, shape=[None], name="td_error")
            self.a_pl = tf.placeholder(tf.int32, shape=[None], name="a")
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden")
            else:
                out = self.s_pl
                for (i, hidden_unit) in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden_unit,
                                          activation=tf.nn.relu, name="hidden_{:d}".format(i))

            out = tf.layers.dense(out, self.action_num, activation=None, name="out")
            self.prob_a = tf.nn.softmax(out)

            with tf.variable_scope("loss"):
                mask = tf.one_hot(self.a_pl, self.action_num, on_value=True, off_value=False)
                prob_log = tf.log(tf.boolean_mask(self.prob_a, mask))
                loss = tf.reduce_mean(-self.td_error_pl*prob_log)
                tf.summary.scalar("loss", loss, collections=["Actor"])
                self.summary = tf.summary.merge_all(key="Actor")

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def learn(self, s, a, td_error, tf_log_writer = None):
        _, summary_val = self.sess.run(
            [self.train_op, self.summary],
            feed_dict={self.s_pl: s, self.a_pl: a, self.td_error_pl: td_error})
        if tf_log_writer:
            tf_log_writer.add_summary(summary_val, self.step_counter)
        self.step_counter += 1

    def choose_action(self, s: np.ndarray):
        if len(s.shape) == 1:
            s = s[np.newaxis, :]
        action_prob = self.sess.run(self.prob_a, feed_dict={self.s_pl: s})
        return np.random.choice(self.actions, p=action_prob.ravel())


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
        with tf.variable_scope("Critic"):
            self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
            self.v_next_pl = tf.placeholder(tf.float32, shape=[None], name="v_")
            self.r_pl = tf.placeholder(tf.float32, shape=[None], name="r")
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden")
            else:
                out = self.s_pl
                for (i, hidden_unit) in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, hidden_unit,
                                          activation=tf.nn.relu, name="hidden_{:d}".format(i))

            self.v = tf.squeeze(tf.layers.dense(out, 1, activation=None, name="out"))

            with tf.variable_scope("loss"):
                self.td_error = self.r_pl + self.gamma * self.v_next_pl - self.v
                self.loss = tf.reduce_mean(tf.square(self.td_error))
                tf.summary.scalar("loss", self.loss, collections=["Critic"])
                tf.summary.histogram("td_error", self.td_error, collections=["Critic"])
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
