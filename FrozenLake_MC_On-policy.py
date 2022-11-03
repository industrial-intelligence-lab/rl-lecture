# On-policy first-visit MC control (p.101)

# TODO 
#

# Imports
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Map setting
map_desc = ["SFFF", "FHFF", "FFFH", "HFFG"]
# map_desc = ["SFFF", "FFFH", "FFFF", "FFFG"]
RENDER_MODE = 'rgb_array'
env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode=RENDER_MODE)

# Constants
NUM_ACTS = env.action_space.n
ETA = 0.1
NUM_EPISODES = 300
GAMMA = 0.9
VERVOSE = False

# Global vars
Q = defaultdict(lambda: np.random.rand(NUM_ACTS))       # {s: [a,...,a]}
Returns = defaultdict(list)                             # {(s, a): [r,...,r]}

# Reporting
tot_rewards = []
tot_steps = []

def random_argmax(value_list):
  """ a random tie-breaking argmax """
  values = np.asarray(value_list)
  return np.argmax(np.random.random(values.shape) * (values==values.max()))

# get action with e-greedy
def get_action(s):
    a = -1
    if np.random.rand() < ETA:
        a = np.random.randint(NUM_ACTS)
        if VERVOSE: print('Random action for %s -> %s' % (s, s))
    else:
        # Find a*
        a = random_argmax(Q[s])
        if VERVOSE: print('Greedy action for %s -> %s' % (s, s))
    return a

# plt.style.use('fivethirtyeight')

for e in range(NUM_EPISODES):
    print("\rEpisode {}/{} --------------------- ".format(e, NUM_EPISODES), end="\n")

    # Env reset
    s, info = env.reset()

    # Terminiation condition
    terminated = False
    truncated = False

    # Trajectory
    traj = []
    num_steps = 0
    fin_reward = 0
    # Generate an episode
    while not terminated and not truncated: 
        # a = np.random.choice(NUM_ACTS, p=PI[s])
        a = get_action(s)
        next_s, r, terminated, truncated, info = env.step(a)       
        traj.append((s, a, r))
        if VERVOSE: print("State:", s, "Action", a, "Reward:", r, "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
        s = next_s
        num_steps += 1
        fin_reward += r
    tot_rewards.append(fin_reward)
    tot_steps.append(num_steps)

    print(f"--> Reward: {fin_reward}, #Steps: {num_steps}, \
                Avg Rewards: {(np.average(tot_rewards))}, Avg steps: {np.average(tot_steps)}")

    # Check trajectory
    if VERVOSE:
        for i, t in enumerate(traj):
            print(i, t)

    # G
    G = 0

    # For each step
    for i, (s, a, r) in reversed(list(enumerate(traj))):
        # Update G
        G = GAMMA*G + r
        first_idx = next(j for j, t in enumerate(traj) if t[0] == s and t[1] == a)
        if VERVOSE: print(i, (s, a, r), first_idx)
        if i == first_idx:
            # print(i, (s, a, r))
            # Append G to (s, a)
            Returns[(s, a)].append(G)
            # Check returns
            if VERVOSE: print('Returns(%s,%s) -> %s' % (s, a, Returns[(s, a)]))            
            # Update Q
            Q[s][a] = sum(Returns[(s, a)]) / len(Returns[(s, a)])
            if VERVOSE: print('Update Q(%s,%s) -> %f' % (s, a, Q[s][a]))

env.close()

# performance graph
def perf_graph(i):    
    plt.cla()
    print()
    plt.plot(range(len(np.array(tot_rewards, dtype=int))), tot_rewards)

if RENDER_MODE != 'human':
    perf_graph(1)
    plt.tight_layout()
    plt.show()