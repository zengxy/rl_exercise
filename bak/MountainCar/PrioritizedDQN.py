import gym
import tensorflow as tf
import numpy as np
import time

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class PrioritiedDQN:
    def __init__(self, actions,
                 learning_rate=0.001, reward_decay=0.9,
                 e_greedy=0.9, e_greedy_increment=0.0002,
                 batch_size=32, hidden_units=10, memory_size=3000,
                 replace_target_iter=300, output_graph=True,
                 double_q=True, prioritized=True):
        self.actions = actions
        self.feature_size = 2

        # RL params
        self.gamma = reward_decay
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.step_counter = 0
        self.double_q = double_q
        self.prioritized = prioritized

        # memory
        self.memory_size = memory_size
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.feature_size * 2 + 2))
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
        if self.prioritized:
            self.ISWeights_pl = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope("eval_net"):
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden")
            else:
                out = self.s_pl
                for i, hidden_unit in enumerate(self.hidden_units):
                    out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                          activation=tf.nn.relu,
                                          name="hidden_{:d}".format(i))
            self.q_predict = tf.layers.dense(out, len(self.actions),
                                             activation=None, name="out")

            with tf.variable_scope("loss"):
                self.abs_errors = tf.reduce_sum(
                    tf.abs(self.q_target_pl - self.q_predict), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(
                    self.ISWeights_pl * tf.squared_difference(self.q_predict, self.q_target_pl))

            with tf.variable_scope("train_op"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # target net: save old params
        self.s_next_pl = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="s_")
        with tf.variable_scope("target_net"):
            if len(self.hidden_units) == 1:
                out = tf.layers.dense(self.s_next_pl, self.hidden_units[0],
                                      activation=tf.nn.relu, name="hidden")
            else:
                out = self.s_pl
                for i, hidden_unit in enumerate(self.hidden_units):
                    out = tf.layers.dense(self.s_pl, self.hidden_units[0],
                                          activation=tf.nn.relu,
                                          name="hidden_{:d}".format(i))
            self.q_target = tf.layers.dense(out, len(self.actions),
                                            activation=None, name="out")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
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

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
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
            q_selected = np.max(q_next_val, 1)

        q_target[batch_index, selected_actions] = r + self.gamma * q_selected

        if self.prioritized:
            _, abs_errors, summary = self.sess.run(
                [self.train_op, self.abs_errors, self.merged_summary],
                feed_dict={self.s_pl: batch_memory[:, :self.feature_size],
                           self.q_target_pl: q_target,
                           self.ISWeights_pl: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
            _, summary = self.sess.run([self.train_op, self.merged_summary],
                                       feed_dict={self.s_pl: batch_memory[:, :self.feature_size],
                                                  self.q_target_pl: q_target})

        if self.file_writer:
            self.file_writer.add_summary(summary, self.step_counter)

        self.epsilon += \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1


def prioritied_dqn_main():
    env = gym.make("MountainCar-v0").unwrapped
    actions = range(0, env.action_space.n)
    rl = PrioritiedDQN(actions=actions, double_q=False, prioritized=True)

    rewards = 0
    episode_i = 0
    total_step = 0
    while True:
        s = env.reset()
        done = False
        step = 0
        while not done:
            a = rl.choose_action(s)
            s_, r, done, _ = env.step(a)
            if not done:
                r = abs(s_[0] - (-0.5)) + abs(s[1])
            rl.store_transition(s, a, r, s_)

            env.render()
            if total_step > 1000:
                rl.learn()
            step += 1
            total_step += 1
            s = s_
        episode_i += 1
        print("Iter: {:d}, finish step: {:d}".format(episode_i, step))
        if episode_i % 100 == 0:
            print("Average reward = {:.3f}".format(rewards))
            rewards = 0


if __name__ == '__main__':
    prioritied_dqn_main()
