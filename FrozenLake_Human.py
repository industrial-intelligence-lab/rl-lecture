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

# For each time step 
while True:    
    # Choose an action from keyboard
    key = readchar.readkey()  
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
 
    # Send a action to env and receive the results
    action = arrow_keys[key]
    state, reward, done, truncated, info= env.step(action)       
    print("State:", state, "Action", action, "Reward:", reward, "Info:", info)

    render = env.render() # Show the board after action as an image
    print('render:',render)

    if done or truncated: 
        print("Finished with reward", reward)
        break

env.close()