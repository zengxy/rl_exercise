import gym
import numpy as np
import tensorflow as tf
from learner.ActorCritic import Actor, Critic
from learner.utils import BasicMemory


def ac_main():
    batch_size = 32
    memory_size = 1000
    env = gym.make("CartPole-v0").unwrapped
    feature_size = env.observation_space.shape[0]
    action_num = env.action_space.n
    sess = tf.Session()
    log_dir = "/tmp/actorCritic"
    memory = BasicMemory(memory_size, feature_size)
    actor = Actor(actions=list(range(action_num)), feature_size=feature_size, sess=sess,
                  hidden_units=20, learning_rate=0.001)
    critic = Critic(feature_size=feature_size, hidden_units=10,
                    sess=sess, learning_rate=0.01, gamma=0.9)
    file_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    total_step = 0
    episode_i = 0
    while True:
        step = 0
        reward = 0
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = actor.choose_action(s)
            s_, r, done, _ = env.step(a)
            if done: r = -20
            reward += r
            transition = np.hstack((s, a, r, s_))
            memory.store(transition)
            if total_step > memory_size:
                s_sample, a_sample, r_sample, s_next_sample = memory.sample(batch_size)
                td_error = critic.learn(s_sample, r_sample, s_next_sample, file_writer)
                if total_step % 50 == 0:
                    actor.learn(s_sample, a_sample, td_error, file_writer)
            s = s_
            step += 1
            total_step += 1
        print("Iter: {:d}, finish step: {:d}, rewards: {:.3f}".format(episode_i, step, reward))
        episode_i += 1


if __name__ == '__main__':
    ac_main()