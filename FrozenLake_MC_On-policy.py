# 참고 https://github.com/djbyrne/MonteCarlo/blob/master/Prediciton.ipynb

# On-policy first-visit MC control (p.101)

# Imports
import gym
import numpy as np
from collections import defaultdict

# Map setting
map_desc = ["SFFF", "FFFH", "FFFH", "HFFG"]
env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode='human')

# Constants
NUM_ACTS = env.action_space.n

# Global vars
PI = defaultdict(lambda: np.ones(NUM_ACTS)/NUM_ACTS)    # [state][action] 
Q = defaultdict(lambda: np.random.rand(NUM_ACTS))       # [state][action]
num_visits = defaultdict(lambda: np.zeros(NUM_ACTS))    # [state][action]
eta = 0.1
num_training = 1

for e in range(num_training):
    # Env reset
    s, info = env.reset()

    # Terminiation condition
    terminated = False
    # Trajectory
    traj = []
    # Generate an episode
    while not terminated: 
        a = np.random.choice(NUM_ACTS, p=PI[s])
        s, r, terminated, truncated, info = env.step(a)       
        traj.append((s, a, r))
        print("State:", s, "Action", a, "Reward:", r, "Terminated:", terminated, "Truncated:", truncated, "Info:", info)

    # Check trajectory
    for i, t in enumerate(traj):
        print(i, t)

    for i, (s, a, r) in enumerate(traj):        
        first_occurence_idx = next(j for j, t in enumerate(traj) if t[0] == s)
        print(s, a, first_occurence_idx)

env.close()