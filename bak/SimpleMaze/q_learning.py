import time
from bak.SimpleMaze.RLBase import RL
from bak.SimpleMaze.const import *
from bak.SimpleMaze.env import Maze


class QLearning(RL):
    def __init__(self, actions, maze_height, maze_width,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearning, self).__init__(actions, maze_height, maze_width, learning_rate,
                                        reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        q_predict = self.q_table.loc[(s.h, s.w), a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[(s_.h, s_.w)].max()
        else:
            q_target = r
        self.q_table.loc[(s.h, s.w), a] += self.lr * (q_target - q_predict)


def q_learning_main():
    maze = Maze(MAZE_HEIGHT, MAZE_WIDTH)
    rl = QLearning(ACTIONS, MAZE_HEIGHT, MAZE_WIDTH)
    for i in range(100):
        s = maze.person
        step = 0
        maze.show()
        print("iter: {}, step:{}".format(i, step))
        time.sleep(0.5)
        while not maze.is_terminal():
            action = rl.choose_action(maze.person)
            s_, r = maze.do_action(action)
            rl.learn(s, action, r, s_)
            s = s_
            step += 1
            maze.show()
            print("iter: {}, step:{}".format(i, step))
            # print(rl.q_table)
            time.sleep(0.5)
        maze.reset()


if __name__ == '__main__':
    q_learning_main()
