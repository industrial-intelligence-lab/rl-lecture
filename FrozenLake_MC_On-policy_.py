# On-policy first-visit MC control (p.101)

# TODO 
#

# Imports
import gym
from Agent_MC_On_policy import MC_On_policy as mc

# Map setting
map_desc = ["SFFF", "FHFF", "FFFH", "HFFG"]
# map_desc = ["SFFF", "FFFH", "FFFF", "FFFG"]

env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode='human')

eng = mc(env, env.action_space.n, 100)
eng.do_train_e_t()
