import gym
import numpy as np
import time
from MountainCar.RLBase import RLBase


class QLearning(RLBase):
    def __init__(self, actions, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self._check_state(s_)
        if s_ == "terminal":
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_].max()
        q_predict = self.q_table.loc[s, a]
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


def make_state(observation):
    position, velocity = observation
    return "{:.2f}, {:.3f}".format(position, velocity)


def q_learning_main():
    env = gym.make("MountainCar-v0").unwrapped
    actions = range(0, env.action_space.n)
    rl = QLearning(actions=actions)

    rewards = 0
    episode_i = 0
    show = False
    while True:
        step = 0
        observation = env.reset()
        s = make_state(observation)
        done = False
        while not done:
            a = rl.choose_action(s)
            observation_, r, done, _ = env.step(a)
            # r = 1000 if done and observation_[0] > -0.4 else np.abs(observation_[1])
            if not done:
                r = r + abs(observation_[0] - (-0.5)) + abs(observation_[1]) - 1
            s_ = make_state(observation_)
            rl.learn(s, a, r, s_)
            s = s_
            if show:
                # print(s, a)
                env.render()
                time.sleep(0.0001)
            # elif step % 1000 == 0:
            #     print(step)
            step += 1
            rewards += r
        if step < 1000:
            show = True
        episode_i += 1
        print("Iter: {:d}, finish step: {:d}".format(episode_i, step))
        if episode_i % 100 == 0:
            print("Average reward = {:.3f}".format(rewards))
            rewards = 0


if __name__ == '__main__':
    q_learning_main()
