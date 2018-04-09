import numpy as np
import os
from collections import namedtuple

from bak.SimpleMaze.const import *

Position = namedtuple("Position", ["h", "w"])


class Maze:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.treasure = Position(np.random.randint(1, height),
                                 np.random.randint(1, width))
        while (self.treasure.h == 1 and self.treasure.w == 1) or \
                (self.treasure.h == height -1 and self.treasure.w == width - 1):
            self.treasure = Position(np.random.randint(1, height),
                                     np.random.randint(1, width))
        self.person = Position(0, 0)
        self.obstacles = [Position(self.treasure.h-1, self.treasure.w),
                          Position(self.treasure.h, self.treasure.w-1)]
        self.obstacles = []

        self.show_map = MAZE_SHOW_MAP
        self.actions = ACTIONS

        # self.maze = self._build()
        self.maze_view = self._build_view()

    def _build(self):
        maze = np.zeros((self.height, self.width))
        maze[self.treasure.h][self.treasure.w] = TREASURE
        maze[self.person.h][self.person.w] = PERSON
        for obstacle in self.obstacles:
            maze[obstacle.h][obstacle.w] = OBSTACLE
        return maze

    def _build_view(self):
        maze_view = \
            [[self.show_map[BLANK] for _ in range(self.width)] for _ in range(self.height)]
        maze_view[self.treasure.h][self.treasure.w] = self.show_map[TREASURE]
        maze_view[self.person.h][self.person.w] = self.show_map[PERSON]
        for obstacle in self.obstacles:
            maze_view[obstacle.h][obstacle.w] = self.show_map[OBSTACLE]
        return maze_view

    def _update_view(self, last_pos):
        self.maze_view[last_pos.h][last_pos.w] = self.show_map[BLANK]
        self.maze_view[self.person.h][self.person.w] = self.show_map[PERSON]

    def show(self):
        os.system('clear')
        for i in range(self.height):
            print(" ".join(self.maze_view[i]))

    def do_action(self, action):
        assert action in self.actions
        last_pos = self.person
        reward = 0
        wall_reward = -10
        if action == "up" or 0:
            if self.person.h > 0:
                self.person = Position(last_pos.h - 1, last_pos.w)
            else:
                reward = wall_reward
        elif action == "down":
            if self.person.h < self.height - 1:
                self.person = Position(last_pos.h + 1, last_pos.w)
            else:
                reward = wall_reward
        elif action == "left":
            if self.person.w > 0:
                self.person = Position(last_pos.h, last_pos.w - 1)
            else:
                reward = wall_reward
        elif action == "right":
            if self.person.w < self.width - 1:
                self.person = Position(last_pos.h, last_pos.w + 1)
            else:
                reward = wall_reward
        self._update_view(last_pos)

        reward = 0
        state = self.person

        if state.h == self.treasure.h and state.w == self.treasure.w:
            reward = 1

        elif any(state.h == obstacle.h and state.w == obstacle.w
                 for obstacle in self.obstacles):
            reward = -1

        return state, reward

    def is_terminal(self):
        if self.person.h == self.treasure.h and self.person.w == self.treasure.w:
            return True
        if any(self.person.h == obstacle.h and self.person.w == obstacle.w
               for obstacle in self.obstacles):
            return True
        return False

    def reset(self, random_set=False):
        # person position to -
        self.maze_view[self.person.h][self.person.w] = self.show_map[BLANK]
        # treasure position to T
        self.maze_view[self.treasure.h][self.treasure.w] = self.show_map[TREASURE]
        for obstacle in self.obstacles:
            self.maze_view[obstacle.h][obstacle.w] = self.show_map[OBSTACLE]
        if not random_set:
            # person to (0, 0)
            self.person = Position(0, 0)
        else:
            while self.is_terminal():
                self.person = Position(
                    np.random.randint(self.height),
                    np.random.randint(self.width))
        self.maze_view[self.person.h][self.person.w] = self.show_map[PERSON]
