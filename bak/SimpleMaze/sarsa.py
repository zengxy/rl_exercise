import time
from bak.SimpleMaze.RLBase import RL
from bak.SimpleMaze.const import *
from bak.SimpleMaze.env import Maze


class SarsaLearning(RL):
    def __init__(self, actions, maze_height, maze_width,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaLearning, self).__init__(actions, maze_height, maze_width, learning_rate,
                                            reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        q_predict = self.q_table.loc[(s.h, s.w), a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[(s_.h, s_.w), a_]
        else:
            q_target = r
        self.q_table.loc[(s.h, s.w), a] += self.lr * (q_target - q_predict)


def sarsa_main():
    maze = Maze(MAZE_HEIGHT, MAZE_WIDTH)
    rl = SarsaLearning(ACTIONS, MAZE_HEIGHT, MAZE_WIDTH)
    for i in range(100):
        s = maze.person
        step = 0
        maze.show()
        print("iter: {}, step:{}".format(i, step))
        time.sleep(0.5)
        a = rl.choose_action(maze.person)
        while not maze.is_terminal():
            s_, r = maze.do_action(a)
            a_ = rl.choose_action(s_)
            rl.learn(s, a, r, s_, a_)
            s = s_
            a = a_
            step += 1
            maze.show()
            print("iter: {}, step:{}".format(i, step))
            # print(rl.q_table)
            time.sleep(0.5)
        maze.reset()


if __name__ == '__main__':
    sarsa_main()
