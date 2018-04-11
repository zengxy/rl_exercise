import gym
from learner.DQN import DeepQNetwork


def dqn_main():
    env = gym.make("CartPole-v0").unwrapped
    actions = range(0, env.action_space.n)
    rl = DeepQNetwork(actions=actions, feature_size=4, hidden_units=10,
                      learning_rate=0.005, e_greedy_increment=0.00005, replace_target_iter=500,
                      memory_size=1000, double_q=False, prioritized=False, dueling=False,
                      output_log_dir="/tmp/DQN-F-F-F")

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