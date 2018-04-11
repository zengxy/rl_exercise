import tensorflow as tf
import numpy as np
from learner.utils import BasicMemory, PrioritizedMemory


class DeepQNetwork:
    def __init__(self, actions, feature_size=2,
                 learning_rate=0.001, reward_decay=0.9,
                 e_greedy=0.9, e_greedy_increment=0.0002,
                 batch_size=32, hidden_units=(10,), memory_size=3000,
                 replace_target_iter=300, output_log_dir=None,
                 double_q=True, prioritized=False, dueling=True):
        self.actions = actions
        self.feature_size = feature_size

        # RL params
        self.gamma = reward_decay
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.step_counter = 0
        self.double_q = double_q
        self.prioritized = prioritized
        self.dueling = dueling

        # memory
        if self.prioritized:
            self.memory = PrioritizedMemory(memory_size, feature_size)
        else:
            self.memory = BasicMemory(memory_size, feature_size)

        # network params
        self.lr = learning_rate
        self.batch_size = batch_size
        self.hidden_units = [hidden_units] if isinstance(hidden_units, int) else hidden_units
        self.replace_target_iter = replace_target_iter
        self._build_network()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "eval_net")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.file_writer = None
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.scalar("loss_value", self.loss)
        self.merged_summary = tf.summary.merge_all()
        if output_log_dir:
            self.file_writer = tf.summary.FileWriter(output_log_dir, self.sess.graph)

    # state ---> DQN ---> action reward
    def _build_network(self):
        self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
        self.q_target_pl = tf.placeholder(tf.float32,
                                          shape=[None, len(self.actions)],
                                          name="q_target")
        if self.prioritized:
            self.ISWeights_pl = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope("eval_net"):
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden",
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer)
            else:
                out = self.s_pl
                for i, hidden_unit in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, self.hidden_units[0],
                                          activation=tf.nn.relu,
                                          name="hidden_{:d}".format(i),
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)
            if self.dueling:
                V = tf.layers.dense(out, 1, activation=None, name="V")
                A = tf.layers.dense(out, len(self.actions), activation=None, name="A")
                out = V + A - tf.reduce_mean(A, axis=1, keep_dims=True)

            self.q_predict = tf.layers.dense(out, len(self.actions),
                                             activation=None, name="out")

        with tf.variable_scope("loss"):
            self.abs_errors = tf.reduce_sum(
                tf.abs(self.q_target_pl - self.q_predict), axis=1)  # for updating Sumtree
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_predict, self.q_target_pl))

        with tf.variable_scope("train_op"):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target net: save old params
        self.s_next_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s_")
        with tf.variable_scope("target_net"):
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_next_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden")
            else:
                out = self.s_next_pl
                for i, hidden_unit in enumerate(self.hidden_units):
                    out = tf.layers.dense(out, self.hidden_units[0],
                                          activation=tf.nn.relu,
                                          name="hidden_{:d}".format(i))
            if self.dueling:
                V = tf.layers.dense(out, 1, activation=None, name="V")
                A = tf.layers.dense(out, len(self.actions), activation=None, name="A")
                out = V + A - tf.reduce_mean(A, axis=1, keep_dims=True)

            self.q_target = tf.layers.dense(out, len(self.actions),
                                            activation=None, name="out")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        self.memory.store(transition)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_predict,
                feed_dict={self.s_pl: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('Target_params_replaced at step {}'.format(self.step_counter))

        if self.prioritized:
            tree_idx, s, a, r, s_, ISWeights = self.memory.sample(self.batch_size)
        else:
            s, a, r, s_ = self.memory.sample(self.batch_size)

        q_predict_val, q_next_val = self.sess.run(
            [self.q_predict, self.q_target],
            feed_dict={
                self.s_pl: s,  # newest params
                self.s_next_pl: s_,  # fixed params
            })

        q_target = q_predict_val.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.double_q:
            q_next_predict_val = self.sess.run(
                self.q_predict, feed_dict={self.s_pl: s_})
            q_selected = q_next_predict_val[batch_index, np.argmax(q_next_val, 1)]
        else:
            q_selected = np.max(q_next_val, axis=1)

        q_target[batch_index, a] = r + self.gamma * q_selected
        _, summary = self.sess.run([self.train_op, self.merged_summary],
                                   feed_dict={self.s_pl: s,
                                              self.q_target_pl: q_target})

        if self.prioritized:
            _, abs_errors, summary = self.sess.run(
                [self.train_op, self.abs_errors, self.merged_summary],
                feed_dict={self.s_pl: s,
                           self.q_target_pl: q_target,
                           self.ISWeights_pl: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
            _, summary = self.sess.run([self.train_op, self.merged_summary],
                                       feed_dict={self.s_pl: s,
                                                  self.q_target_pl: q_target})

        if self.file_writer:
            self.file_writer.add_summary(summary, self.step_counter)

        self.epsilon += \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1
