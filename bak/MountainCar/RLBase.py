import numpy as np
import pandas as pd


class RLBase:
    def __init__(self, actions,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float32)

    def _check_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))

    def choose_action(self, state):
        self._check_state(state)
        state_actions = self.q_table.loc[state]
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # 找出所有的max id,
            max_value = state_actions.max()
            all_max_id = state_actions.index[state_actions == max_value].tolist()
            action = np.random.choice(all_max_id)
        return action

    def learn(self, *args, **kwargs):
        pass
