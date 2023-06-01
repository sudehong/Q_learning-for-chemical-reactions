import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# 打开一个 pickle 文件并读取其中的数据
with open('Pss_mat.pickle', 'rb') as f:
    data = pickle.load(f)

# d1 = data[0]
d2 = data[1]
print(d2)


#
l = len(d2)
# print(l)

class Enviroment():

    def __init__(self):
        self.states = [i for i in range(l)]
        self.goal_state = 451
        self.start_state = 0
        self.agent_state = self.start_state

    def reward(self, state, next_state):
        if next_state == self.goal_state:
            return 1
        else:
            return 0

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    #action = next_state
    def step(self,state,action):

        next_state = action
        reward = self.reward(state,next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state,p,act_dict):
        action_probs = {key:p[state] for key in act_dict[state].keys()}
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in d2[next_state].keys()]
            next_q_max = max(next_qs)

        target = (reward + self.gamma * next_q_max)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

# act_dict = [list(d2.values()) for key in d2.keys()]
# print(d2)
env = Enviroment()
agent = QLearningAgent()

def calculate_row_averages(d):
    averages_dict = {}
    # print(d)
    for i in d:
        sum_row = sum(d[i].values())
        if sum_row:
            row_average = 1 / sum_row
        else:
            row_average = 0
        averages_dict[i] = row_average
    return averages_dict


p = calculate_row_averages(d2)


episodes = 1000
Q = 0

for episode in tqdm(range(episodes)):
    state = env.reset()

    while True:
        action = agent.get_action(state,p,d2)
        next_state, reward, done = env.step(state,action)
        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

print(agent.Q)