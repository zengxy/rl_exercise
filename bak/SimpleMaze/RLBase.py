import numpy as np
import pandas as pd
from typing import Union
from bak.SimpleMaze.env import Position


class RL:
    def __init__(self, actions, maze_height, maze_width,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.maze_height = maze_height
        self.maze_width = maze_width
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = self._build_q_table()

    def _build_q_table(self):
        idx = pd.MultiIndex.from_product(
            [range(self.maze_height), range(self.maze_width)], names=["h", "w"])
        table = pd.DataFrame(
            np.zeros((self.maze_height*self.maze_width, len(self.actions))),
            index=idx, columns=self.actions)
        return table

    def choose_action(self, position: Union[Position, str]):
        if position == "terminal":
            return "no action"
        state_actions = self.q_table.loc[(position.h, position.w), :]
        if np.random.uniform() > self.epsilon or (state_actions == 0).all():
            action = np.random.choice(self.actions)
        else:
            # 找出所有的max id,
            max_value = state_actions.max()
            all_max_id = state_actions.index[state_actions == max_value].tolist()
            action = np.random.choice(all_max_id)
        return action

    def learn(self, *args, **kwargs): pass
