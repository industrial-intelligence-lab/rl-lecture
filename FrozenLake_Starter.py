# On-policy first-visit MC control (p.101)

# Imports
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from Agent_MC_On_policy import MC_On_policy as mc
from Agent_Sarsa_On_policy import Sarsa_On_policy as sa

# figure


# Map setting
map_desc = ["SFFF", "FHFF", "FFFH", "HFFG"]
# map_desc = ["SFFF", "FFFH", "FFFF", "FFFG"]

# env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode='rgb_array') #'human')
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='rgb_array') #'human')

start = time.time()
rewards = []
for i in range(100):
    eng = mc(env, env.action_space.n, 500, ETA=0.1, VERVOSE=False, REPORTING=False)
    r = eng.do_train_e_t()
    rewards.append(r)
end = time.time()
print(f"{end - start:.5f} sec")
y = np.average(np.array(rewards), axis=0)
x = range(len(y))
plt.plot(x, y, label='mc')

start = time.time()
rewards = []
for i in range(100):
    eng = sa(env, env.action_space.n, 500, ETA=0.1, ALPHA=0.2, VERVOSE=False, REPORTING=False)
    r = eng.do_train_e_t()
    rewards.append(r)
end = time.time()
print(f"{end - start:.5f} sec")
y = np.average(np.array(rewards), axis=0)
x = range(len(y))
plt.plot(x, y, label='sarsa')

plt.legend()
plt.show()