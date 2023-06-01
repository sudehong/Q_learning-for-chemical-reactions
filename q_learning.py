import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from tqdm import tqdm



import numpy as np
from fractions import Fraction
import random
from built import draw_pitures
from collections import defaultdict
import copy

def find_negative_elements(matrix):
    matrix = np.array(matrix)
    negative_elements = matrix[matrix < 0]
    return negative_elements.tolist()

N = 5

d = [4,3,2,1]

def find_matrix(N,d):


    list_num = sum(range(1, N-1))

    # list n2ij : index of matrix
    n2ij = []
    index = 0
    for i in range(N-1):
        for j in range(i+1,N-1):
            n2ij.append([i,j])
            index += 1

    n2ij_original = n2ij.copy()
    n2ij.reverse()

    kk = []
    def generate_lists(n, prefix=[]):

        if n == 0:
            # prefix to  matrix
            m = np.zeros((N, N))
            for j in range(len(prefix)):
                k = n2ij_original[j]
                row = k[0]
                col = k[1]
                m[row][col] = m[col][row] = prefix[j]

            #填充最后一列
            for j in range(N-1):
                m[j][N-1] =  d[j] - sum(m[j,:N-1])
            #最后一行
            for j in range(N-1):
                m[N-1][j] = d[j] - sum(m[:N-1,j])

            if m[m<0].tolist():
                pass
            else:
                kk.append(m)
            return
        matrix = np.zeros((N, N))
        #列表部不为空时，填充matrix
        if len(prefix) != 0:
            # l = len(prefix)
            for j in range(len(prefix)):
                k = n2ij_original[j]
                row = k[0]
                col = k[1]
                matrix[row][col] = matrix[col][row] = prefix[j]
        else:
            matrix[0][0] = 0
        #根据矩阵确定当前数的上限
        #根据n,取当前要确定数的矩阵坐标
        k1 = n2ij[n-1]
        row1 = k1[0]
        col1 = k1[1]
        #根据列，取前面行元素的和,并取最大值
        if row1 !=0 :
            #取行,列
            h = matrix[row1,:col1]
            l = matrix[0:row1,col1]
            maximum1 = max(sum(h),sum(l))
        else:
            #在第一行时，只需取前面行
            h = matrix[row1, :col1]
            maximum1 = sum(h)

        for i in range(0, d[row1]-int(maximum1)+1):
            generate_lists(n - 1, prefix+[i])

    generate_lists(list_num)

    return kk


# print(find_matrix(N,d))
# print(len(kk))

kk = find_matrix(N,d)

l = len(kk)
#Pss'
Pss_matrix = np.zeros((l,l))

def check_list(lst):
    count_1 = 0
    count_minus_1 = 0

    for num in lst:
        if num == 1:
            count_1 += 1
        elif num == -1:
            count_minus_1 += 1
        elif num != 0:
            return False

    return (count_1 == 2 and count_minus_1 == 0) or (count_1 == 0 and count_minus_1 == 2)

def get_indexes(lst):
    indexes = []
    for i, num in enumerate(lst):
        if num == 1 or num == -1:
            indexes.append(i)
    return indexes


#比较两个两个矩阵
for i in range(l):
    for j in range(l):
        diffence = kk[i] - kk[j]
        #判断最后一列元素是否有非1,-1元素
        if check_list(diffence[:,N-1]):
            #再次判断这个两个1或者-1的下标，对应的位置上是否是相反数
            index = get_indexes(diffence[:,N-1])
            if abs(diffence[index[0]][index[1]]) == 1 and sum(abs(n) for n in diffence[index[0],:]) == 2 and sum(abs(n) for n in diffence[index[1],:]) == 2:
                Pss_matrix[i][j] = 1
                # draw_pitures(kk[i], kk[j], i, j, 1)
            else:
                Pss_matrix[i][j] = 0
                # draw_pitures(kk[i], kk[j], i, j, 0)
        else:
            Pss_matrix[i][j] = 0
            # draw_pitures(kk[i], kk[j], i, j, 0)

print(Pss_matrix)

def find_indexes(lst, target):
    indexes = []
    for i, element in enumerate(lst):
        if element == target:
            indexes.append(i)
    return indexes


def calculate_row_averages(matrix):
    averages_dict = {}
    for i, row in enumerate(matrix):
        non_zero_elements = [element for element in row if element != 0]
        if non_zero_elements:
            row_average = 1 / len(non_zero_elements)
        else:
            row_average = 0
        averages_dict[i] = row_average
    return averages_dict



def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


class Enviroment():

    def __init__(self,Pss_matrix):
        self.states = [i for i in range(61)]
        self.action = [i for i in range(61)]
        self.goal_state = 12
        self.start_state = 0
        # self.r = np.zeros((61,61))
        # self.r[:, self.goal_state] = Pss_matrix[:, self.goal_state]
        # self.reward_map = self.r
        self.agent_state = self.start_state

    def reward(self, state, next_state):
        if next_state == self.goal_state:
            return 1
        else:
            return 0


    def actions(self):
        return self.action

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
        self.epsilon = 0.1
        # self.action = act_dict
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state,p,act_dict):
        action_probs = {key:p[state] for key in act_dict[state]}
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in act_dict[next_state]]
            next_q_max = max(next_qs)

        target = (reward + self.gamma * next_q_max)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        # self.b[state] = greedy_probs(self.Q, state, self.epsilon,act_dict[state])


act_dict = {key: find_indexes(Pss_matrix[key,:],1) for key in range(0, 61)}
env = Enviroment(Pss_matrix)
agent = QLearningAgent()



p = calculate_row_averages(Pss_matrix.tolist())
# print(p)
print(act_dict)
#
# for state in range(len(p)):
#     print({key:p[state] for key in act_dict[state]})

episodes = 1000
Q = 0

for episode in tqdm(range(episodes)):
    state = env.reset()

    while True:
        action = agent.get_action(state,p,act_dict)
        next_state, reward, done = env.step(state,action)
        agent.update(state, action, reward, next_state, done)
        if done:
            Q = agent.Q
            break
        state = next_state


print(agent.Q)



