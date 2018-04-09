import time
from bak.SimpleMaze.RLBase import RL
from bak.SimpleMaze.const import *
from bak.SimpleMaze.env import Maze


class SarsaLambdaLearning(RL):
    def __init__(self, actions, maze_height, maze_width,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay = 0.9):
        super(SarsaLambdaLearning, self).__init__(actions, maze_height, maze_width, learning_rate,
                                                  reward_decay, e_greedy)
        self.e_table = self.q_table.copy()
        self.lambda_ = trace_decay

    def learn(self, s, a, r, s_, a_):
        q_predict = self.q_table.loc[(s.h, s.w), a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[(s_.h, s_.w), a_]
        else:
            q_target = r
        error = q_target - q_predict
        self.e_table.loc[(s.h, s.w), a] += 1
        self.q_table += self.lr * error * self.e_table

        self.e_table.loc[(s.h, s.w), a] *= self.gamma * self.lambda_


def sarsa_lambda_main():
    maze = Maze(MAZE_HEIGHT, MAZE_WIDTH)
    rl = SarsaLambdaLearning(ACTIONS, MAZE_HEIGHT, MAZE_WIDTH)
    for i in range(100):
        s = maze.person
        rl.e_table *= 0
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
            # input("press any key to continue")
        maze.reset()


if __name__ == '__main__':
    sarsa_lambda_main()
