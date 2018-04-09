import gym
import tensorflow as tf
import numpy as np
import time


class DeepQNetwork:
    def __init__(self, actions, feature_size=2,
                 learning_rate=0.001, reward_decay=0.9,
                 e_greedy=0.9, e_greedy_increment=0.0002,
                 batch_size=32, hidden_units=(10,), memory_size=3000,
                 replace_target_iter=300, output_graph=True,
                 double_q=True):
        self.actions = actions
        self.feature_size = feature_size

        # RL params
        self.gamma = reward_decay
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.step_counter = 0
        self.double_q = double_q

        # memory
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, 2 * self.feature_size + 2))  # memory: s, a, r, s_
        self.memory_counter = 0

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
        self.merged_summary = None
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if output_graph:
            tf.summary.scalar("loss_value", self.loss)
            self.merged_summary = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter("/tmp/DQN/", self.sess.graph)

    # state ---> DQN ---> action reward
    def _build_network(self):
        self.s_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s")
        self.q_target_pl = tf.placeholder(tf.float32,
                                          shape=[None, len(self.actions)],
                                          name="q_target")
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
            self.q_predict = tf.layers.dense(out, len(self.actions),
                                             activation=None, name="out")

        with tf.variable_scope("loss"):
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
            self.q_target = tf.layers.dense(out, len(self.actions),
                                            activation=None, name="out")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

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

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_predict_val, q_next_val = self.sess.run(
            [self.q_predict, self.q_target],
            feed_dict={
                self.s_pl: batch_memory[:, :self.feature_size],  # newest params
                self.s_next_pl: batch_memory[:, -self.feature_size:],  # fixed params
            })

        q_target = q_predict_val.copy()
        selected_actions = batch_memory[:, self.feature_size].astype(int)
        r = batch_memory[:, self.feature_size + 1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.double_q:
            q_next_predict_val = self.sess.run(
                self.q_predict, feed_dict={self.s_pl: batch_memory[:, -self.feature_size:]})
            q_selected = q_next_predict_val[batch_index, np.argmax(q_next_val, 1)]
        else:
            q_selected = np.max(q_next_val, axis=1)

        q_target[batch_index, selected_actions] = r + self.gamma * q_selected
        _, summary = self.sess.run([self.train_op, self.merged_summary],
                                   feed_dict={self.s_pl: batch_memory[:, :self.feature_size],
                                              self.q_target_pl: q_target})

        if self.file_writer:
            self.file_writer.add_summary(summary, self.step_counter)

        self.epsilon += \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1


def dqn_main():
    env = gym.make("CartPole-v1").unwrapped
    actions = range(0, env.action_space.n)
    rl = DeepQNetwork(actions=actions, feature_size=4, double_q=True, hidden_units=[12, 6],
                      learning_rate=0.005, e_greedy_increment=0.00005, replace_target_iter=500,
                      memory_size=1000)

    episode_i = 0
    total_step = 0
    while True:
        s = env.reset()
        done = False
        step = 0
        rewards = 0
        while not done:
            a = rl.choose_action(s)
            s_, r, done, _ = env.step(a)
            # if done:
            #     r = 100
            # r = abs(s_[0] - (-0.5)) + abs(s[1])
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            # r = r1+r2
            rewards += r
            rl.store_transition(s, a, r, s_)
            env.render()
            # time.sleep(0.01)
            if total_step > 1000:
                rl.learn()
            step += 1
            total_step += 1
            s = s_
        episode_i += 1
        print("Iter: {:d}, finish step: {:d}, rewards: {:.3f}".format(episode_i, step, rewards))


if __name__ == '__main__':
    dqn_main()
