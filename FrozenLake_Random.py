# Imports
import gym
import readchar

# Key mapping
arrow_keys = {
    'w' : 3, # UP
    's' : 1, # DOWN
    'd' : 2, #RIGHT
    'a' : 0, #LEFT
}

# Map setting
map_desc = ["SFFF", "FFFH", "FFFH", "HFFG"]
env = gym.make('FrozenLake-v1', desc=map_desc, map_name="4x4", is_slippery=False, render_mode='human')

# Env reset
observation, info = env.reset()

# terminiation condition
terminated = False
# For each time step 
while not terminated: 
    # Send a random action to env and receive the results
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)       
    print("State:", state, "Action", action, "Reward:", reward, "Terminated:", terminated, "Truncated:", truncated, "Info:", info)

    render = env.render() # Show the board after action as an image
    print('render:',render)

env.close()