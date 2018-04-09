MAZE_HEIGHT = 4
MAZE_WIDTH = 4
ACTIONS = ["up", "down", "left", "right"]

EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9     # discount factor

BLANK = 0
OBSTACLE = -1
TREASURE = 1
PERSON = 2
MAZE_SHOW_MAP = {BLANK: "-", OBSTACLE: "*", TREASURE: "T", PERSON: "o"}
