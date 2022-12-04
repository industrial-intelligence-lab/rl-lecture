# On-policy first-visit MC control (p.101)

# Imports
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from Agent_MC_On_policy import MC_On_policy as mc
from Agent_Sarsa_On_policy import Sarsa_On_policy as sa
from Agent_Qlearning_On_policy import Qlearning_On_policy as q
from Agent_DQN_On_policy import DQN_On_policy as dqn


# Map setting
map_desc = ["SFFF", "FHFF", "FFFH", "HFFG"]
# map_desc = ["SFFF", "FFFH", "FFFF", "FFFG"]

env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode='rgb_array') #'rgb_array') #'human')
# env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='rgb_array') #'human')

NUM_RUNS = 1
NUM_EPISODES = 300

start = time.time()
rewards = []
for i in range(NUM_RUNS):
    print(i)
    eng = dqn(env, env.action_space.n, NUM_EPISODES, ETA=0.9, ETA_DELTA=0.005, ETA_MIN=0.001, VERVOSE=True, REPORTING=False)
    r = eng.do_train_e_t()
    rewards.append(r)
end = time.time()
print(f"{end - start:.5f} sec")
y = np.average(np.array(rewards), axis=0)
x = range(len(y))
plt.scatter(list(x), y, s=0.2)

# start = time.time()
# rewards = []
# for i in range(NUM_RUNS):
#     print(i)
#     eng = sa(env, env.action_space.n, NUM_EPISODES, ETA=0.05, ALPHA=0.1, VERVOSE=False, REPORTING=False)
#     r = eng.do_train_e_t()
#     rewards.append(r)
# end = time.time()
# print(f"{end - start:.5f} sec")
# y = np.average(np.array(rewards), axis=0)
# x = range(len(y))
# plt.plot(x, y, label='sarsa')

plt.legend()
plt.show()