import time
from bak.SimpleMaze.const import *
from bak.SimpleMaze.env import Maze
import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(self, actions,
                 learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.9, e_greedy_increment=None,
                 batch_size=64, hidden_units=8, memory_size=500,
                 replace_target_iter=300, output_graph=True,
                 double_q=True):
        self.actions = actions
        self.feature_size = 2

        # RL params
        self.gamma = reward_decay
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.step_counter = 0
        self.double_q = double_q

        # memory
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, 2*self.feature_size + 2))  # memory: s, a, r, s_
        self.memory_counter = 0
        self.memory_test = np.zeros((self.memory_size, 2*self.feature_size + 3))

        # network params
        self.lr = learning_rate
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.replace_target_iter = replace_target_iter
        self._build_network()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
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
        normal_initializer = tf.random_normal_initializer()
        zero_initializer = tf.zeros_initializer()
        self.s_pl = tf.placeholder(tf.float32, shape=[None, 2], name="position")
        self.q_target_pl = tf.placeholder(tf.float32,
                                          shape=[None, len(self.actions)],
                                          name="q_target")
        with tf.variable_scope("eval_net"):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope("hidden"):
                w = tf.get_variable("w", shape=[self.feature_size, self.hidden_units],
                                    initializer=normal_initializer,
                                    collections=c_names)
                b = tf.get_variable("b", shape=[self.hidden_units],
                                    initializer=zero_initializer,
                                    collections=c_names)
                h1 = tf.nn.relu(tf.matmul(self.s_pl, w) + b)

            with tf.variable_scope("out"):
                w = tf.get_variable("w", shape=[self.hidden_units, len(self.actions)],
                                    initializer=normal_initializer,
                                    collections=c_names)
                b = tf.get_variable("b", shape=[len(self.actions)],
                                    initializer=zero_initializer,
                                    collections=c_names)
                self.q_predict = tf.matmul(h1, w) + b

            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_predict, self.q_target_pl))

            with tf.variable_scope("train_op"):
                self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # target net: save old params
        self.s_next_pl = tf.placeholder(tf.float32, shape=[None, 2], name="position")
        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope("hidden"):
                w = tf.get_variable("w", shape=[self.feature_size, self.hidden_units],
                                    initializer=normal_initializer,
                                    collections=c_names)
                b = tf.get_variable("b", shape=[self.hidden_units],
                                    initializer=zero_initializer,
                                    collections=c_names)
                h1 = tf.nn.relu(tf.matmul(self.s_next_pl, w) + b)

            with tf.variable_scope("out"):
                w = tf.get_variable("w", shape=[self.hidden_units, len(self.actions)],
                                    initializer=normal_initializer,
                                    collections=c_names)
                b = tf.get_variable("b", shape=[len(self.actions)],
                                    initializer=zero_initializer,
                                    collections=c_names)
                self.q_target = tf.matmul(h1, w) + b

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.h, s.w, a, r, s_.h, s_.w))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_test[index, :] = np.hstack((s.h, s.w, a, r, s_.h, s_.w, self.memory_counter))

        self.memory_counter += 1

    def choose_action(self, position):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_predict,
                                          feed_dict={
                                              self.s_pl: np.array([[position.h, position.w]])})
            action = np.argmax(actions_value)
        else:
            action = np.random.choice(len(self.actions))
        return action

    def learn(self):
        # check to replace target parameters
        if self.step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('Target_params_replaced at step {}'.format(self.step_counter))
            time.sleep(2)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_predict_val, q_next_val = self.sess.run(
            [self.q_predict, self.q_target],
            feed_dict={
                self.s_pl: batch_memory[:, :self.feature_size],        # newest params
                self.s_next_pl: batch_memory[:, -self.feature_size:],  # fixed params
            })

        q_target = q_predict_val.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.feature_size].astype(int)
        reward = batch_memory[:, self.feature_size+1]

        if self.double_q:
            q_next_predict_val = self.sess.run(
                self.q_predict,
                feed_dict={self.s_pl: batch_memory[:, -self.feature_size:]})
            q_selected = q_next_predict_val[batch_index, np.argmax(q_next_val, axis=1)]
        else:
            q_selected = np.max(q_next_val, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * q_selected

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, summary = self.sess.run([self.train_op, self.merged_summary],
                                   feed_dict={self.s_pl: batch_memory[:, :self.feature_size],
                                              self.q_target_pl: q_target})
        self.file_writer.add_summary(summary, self.step_counter)
        # increasing epsilon
        self.epsilon += \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def dqn_main():
    maze = Maze(MAZE_HEIGHT, MAZE_WIDTH)
    rl = DeepQNetwork(actions=ACTIONS,
                      maze_height=MAZE_HEIGHT,
                      maze_width=MAZE_WIDTH,
                      e_greedy_increment=0.01)
    step = 0
    for i in range(10000):
        s = maze.person
        maze.show()
        a = rl.choose_action(s)
        print("iter: {}, step:{}, action:{}".format(i, step, ACTIONS[a]))
        time.sleep(0.5)
        random_set = True
        while not maze.is_terminal():
            a = rl.choose_action(s)
            s_, r = maze.do_action(action=ACTIONS[a])
            rl.store_transition(s, a, r, s_)
            if step > 300 and step % 5 == 0:
                rl.learn()
                print("learning......")

            if step > 1000:
                random_set = False
                maze.show()
                print("iter: {}, step:{}, action:{}".format(i, step, ACTIONS[a]))
                time.sleep(0.5)
            s = s_

            step += 1
        maze.reset(random_set=random_set)


if __name__ == '__main__':
    dqn_main()
