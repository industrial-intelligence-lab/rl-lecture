# imports from home at two
import gym
import readchar

#MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
 
# Key mapping
arrow_keys = {
    'w' : UP,
    's' : DOWN,
    'd' : RIGHT,
    'a' : LEFT
}
 
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
state = env.reset()

while True:    
    # Choose an action from keyboard
    key = readchar.readkey()  
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
 
    action = arrow_keys[key]
    print(key, arrow_keys[key])
    state, reward, done, truncated, info= env.step(action)       

    env.render() # Show the board after action
    print("State:", state, "Action", action, "Reward:", reward, "Info:", info)
 
    if done: 
        print("Finished with reward", reward)
        break
