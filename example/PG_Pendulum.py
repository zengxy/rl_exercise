import gym
import numpy as np
import tensorflow as tf
from learner.PolicyGradient_Continuous import PolicyGradient
from learner.utils import BasicMemory


def pg_main():
    max_step_ep = 100
    env = gym.make("Pendulum-v0").unwrapped
    feature_size = env.observation_space.shape[0]
    action_high = env.action_space.high[0]
    log_dir = "/tmp/pg_cont"
    rl = PolicyGradient(-action_high, action_high, feature_size, tf_log_dir=log_dir)
    # sess.run(tf.global_variables_initializer())
    total_step = 0
    episode_i = 0
    while True:
        step = 0
        reward = 0
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = rl.choose_action(s)
            s_, r, done, _ = env.step(a)
            r /= 10
            rl.store_transition(s, a, r)
            if done or step>max_step_ep:
                rl.learn()
                done = True
            reward += r
            s = s_
            step += 1
            total_step += 1
        print("Iter: {:d}, finish step: {:d}, rewards: {:.3f}".format(episode_i, step, reward))
        episode_i += 1


if __name__ == '__main__':
    pg_main()