# Deep Q Learning / Frozen Lake / Not Slippery / 4x4
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
# warnings.filterwarnings("ignore")
import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

custom_map = [
    'FFFF',
    'FHFH',
    'FFFH',
    'FHFG'
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode='rgb_array')
train_episodes=500
test_episodes=10
max_steps=100
state_size = env.observation_space.n
action_size = env.action_space.n
# batch_size=32

class Agent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2500)
        self.learning_rate=0.01
        self.epsilon=1
        self.max_eps=1
        self.min_eps=0.01
        self.eps_decay = 0.01#/3
        self.gamma=0.9
        self.state_size= state_size
        self.action_size= action_size
        self.epsilon_lst=[]
        self.model = self.buildmodel()

    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values==values.max()))

    def buildmodel(self):
        self.optimizers = optimizers.Adam(lr=0.01, )

        model=Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='linear'))
        model.add(Dense(32, input_dim=self.state_size, activation='linear'))
        model.add(Dense(self.action_size, activation='linear'))
        return model

    def add_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0,4)
        ppp = self.model.predict(state)
        # print('predict', ppp, state)
        return self.random_argmax(ppp)

    def pred(self, state):
        return self.random_argmax(self.model.predict(state))

    # def replay(self,batch_size):
    #     print('replay~!!!!!!!!!!!!!!!!!')
    #     minibatch=random.sample(self.memory, batch_size)
    #     for new_state, reward, done, state, action in minibatch:
    #         target= reward
    #         if not done:
    #             target=reward + self.gamma* np.amax(self.model.predict(new_state))
    #         target_f= self.model.predict(state)
    #         target_f[0][action]= target
    #         self.model.fit(state, target_f, epochs=1, verbose=1)

    #     # if self.epsilon > self.min_eps:
    #     #     self.epsilon=(self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

    #     self.epsilon_lst.append(self.epsilon)

    def train(self, state, action, reward, new_state, done):
        dqn_variable = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            curr_Q = self.model([state])[0]
            ## Obtain the Q' values by feeding the new state through our network
            next_Q = np.asarray(self.model([new_state]))

            ## Obtain maxQ' and set our target value for chosen action.
            q_target = np.array(curr_Q)

            # But from target model
            if done:
                q_target[action] = reward
            else:
                q_target[action] = (reward + self.gamma * np.max(next_Q[0]))
            
            ## Train network using target and predicted Q values
            # it is not real target Q value, it is just an estimation,
            # but check the Q-Learning update formula:
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a) - Q(s,a)]
            # minimizing |r + gamma * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ~~ Q(s,a)            
            q_value = self.model([state])
            # print('q_value',q_value)
          
            main_value   = tf.convert_to_tensor(q_value)
            target_value = tf.convert_to_tensor(q_target)
            error = tf.square(main_value - target_value) * 0.5
            loss  = tf.reduce_mean(error)
            
        dqn_grads = tape.gradient(loss, dqn_variable)
        self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent=Agent(state_size, action_size)

reward_lst=[]
for episode in range(train_episodes):
    print('episode', episode)
    state= env.reset()[0]
    state_arr=np.zeros(state_size)
    state_arr[state] = 1
    state= np.reshape(state_arr, [1, state_size])
    reward = 0
    done = False
    for t in range(max_steps):
        # env.render()
        action = agent.action(state)
        # print(state, action)
        # new_state, reward, done, info = env.step(action)
        new_state, reward, done, truncated, info = env.step(action)       
        # print(reward)
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        # agent.add_memory(state, action, reward, new_state, done)
        agent.train(state, action, reward, new_state, done)
        
        state= new_state
    
        if done:
            print(f'Episode: {episode:4}/{train_episodes} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}')
            if agent.epsilon > agent.min_eps:
                agent.epsilon=(agent.max_eps - agent.min_eps) * np.exp(-agent.eps_decay*episode) + agent.min_eps
                agent.epsilon_lst.append(agent.epsilon)
            break

    reward_lst.append(reward)
    
    # if len(agent.memory)> batch_size:
    #     agent.replay(batch_size)

print(' Train mean % score= ', round(100*np.mean(reward_lst),1))

# test
test_wins=[]
env.close()
env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode='human')

for episode in range(test_episodes):
    state = env.reset()[0]
    state_arr=np.zeros(state_size)
    state_arr[state] = 1
    state= np.reshape(state_arr, [1, state_size])
    done = False
    reward=0
    state_lst = []
    state_lst.append(state)
    print('******* EPISODE ',episode, ' *******')

    for step in range(max_steps):
        action = agent.pred(state)
        new_state, reward, done, truncated, info = env.step(action)       
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        state = new_state
        state_lst.append(state)
        if done:
            print(reward)
            # env.render()
            break
    print('reward',reward)
    test_wins.append(reward)
env.close()

print(' Test mean % score= ', int(100*np.mean(test_wins)))

fig=plt.figure(figsize=(10,12))
matplotlib.rcParams.clear()
matplotlib.rcParams.update({'font.size': 22})
plt.subplot(311)
plt.scatter(list(range(len(reward_lst))), reward_lst, s=0.2)
plt.title('4x4 Frozen Lake Result(DQN) \n \nTrain Score')
plt.ylabel('Score')
plt.xlabel('Episode')

plt.subplot(312)
plt.scatter(list(range(len(agent.epsilon_lst))), agent.epsilon_lst, s=0.2)
plt.title('Epsilon')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

plt.subplot(313)
plt.scatter(list(range(len(test_wins))), test_wins, s=0.5)
plt.title('Test Score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.ylim((0,1.1))
plt.savefig('result.png',dpi=300)
plt.show()