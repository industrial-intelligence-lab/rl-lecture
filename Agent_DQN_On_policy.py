import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

class DQN_On_policy:

    def __init__(self, env, NUM_ACTS, NUM_EPISODES=100, ETA=0.9, ETA_DELTA=0.01, ETA_MIN=0.001, GAMMA=0.9, ALPHA=0.1, VERVOSE=True, REPORTING=True):
        self.env = env
        self.NUM_ACTS = NUM_ACTS
        self.NUM_EPISODES = NUM_EPISODES
        self.ETA = ETA
        self.ETA_DELTA = ETA_DELTA
        self.ETA_MIN = ETA_MIN
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.VERVOSE = VERVOSE
        self.REPORTING = REPORTING
        self.NUM_STATES = env.observation_space.n
        self.model = self.buildmodel()
        self.memory = deque(maxlen=2048)
        self.batch_size = 64

    def buildmodel(self):
        self.optimizers = optimizers.Adam(lr=0.01, )

        activation = 'linear'
        model=Sequential()
        model.add(Dense(128, input_dim=self.NUM_STATES, activation=activation))
        model.add(Dense(64, activation=activation))
        model.add(Dense(32, activation=activation))
        model.add(Dense(self.NUM_ACTS, activation=activation))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values==values.max()))

    # get action with e-greedy
    def get_action(self, s):
        a = -1
        if np.random.rand() < self.ETA:
            a = np.random.randint(self.NUM_ACTS)
            # if self.VERVOSE: print('Random action for %s -> %s' % (s, a))
        else:
            q = self.model.predict(s)
            a = np.argmax(q)
            # if self.VERVOSE: print('Greedy action for %s -> %s' % (s, a))
        return a

    def get_one_hot(self, s):
        state_arr=np.zeros(self.NUM_STATES)
        state_arr[s] = 1
        state= np.reshape(state_arr, [1, self.NUM_STATES])
        return state

    def train_using_memory(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for s, a, r, next_s, terminated in minibatch:
            q_value = self.model([s])
            target_q = np.array(q_value)
            next_q = np.array(self.model([next_s]))
            if terminated:
                target_q[0][a] = r
            else:
                target_q[0][a] = r + self.GAMMA * np.max(next_q[0])
            print('training -> ', s, target_q)
            self.model.fit(s, target_q, epochs=1, verbose=0)

        if self.ETA - self.ETA_DELTA > self.ETA_MIN:
            self.ETA = self.ETA - self.ETA_DELTA
            
    # def train(self, s, a, r, next_s, terminated):
    #     # weights for training
    #     dqn_variable = self.model.trainable_variables
    #     with tf.GradientTape() as tape:
    #         tape.watch(dqn_variable)
    #         # q for s
    #         q_value = self.model([s])
    #         target_q = np.array(q_value[0])
    #         next_q = np.array(self.model([next_s]))

    #         if terminated:
    #             target_q[a] = r
    #         else:
    #             target_q[a] = r + self.GAMMA * np.max(next_q[0])

    #         main_value   = tf.convert_to_tensor(q_value)
    #         target_value = tf.convert_to_tensor(target_q)
    #         error = tf.square(target_value - main_value) * 0.5
    #         loss  = tf.reduce_mean(error)             
      
    #     dqn_grads = tape.gradient(loss, dqn_variable)
    #     self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))

    # 전체 episode와 time step을 반복하면서 학습진행->종료
    def do_train_e_t(self):
        # Reporting
        reward_list = []
        eta_list = []

        for e in range(self.NUM_EPISODES):
            if self.VERVOSE: print("\rEpisode {}/{} --------------------- ".format(e, self.NUM_EPISODES), end="\n")

            # Env reset
            s, info = self.env.reset()
            s = self.get_one_hot(s)

            # Terminiation condition
            terminated = False
            truncated = False

            num_steps = 0
            fin_reward = 0
            # Generate an episode
            while not terminated and not truncated: 
                a = self.get_action(s)
                next_s, r, terminated, truncated, info = self.env.step(a)   
                next_s = self.get_one_hot(next_s)
                # self.train(s, a, r, next_s, terminated)
                self.memory.append((s, a, r, next_s, terminated))

                if self.VERVOSE: print("State:", s, "Action", a, "Reward:", r, "Next state:", next_s, \
                                                     "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
                
                s = next_s
                num_steps += 1
                fin_reward += r

            if len(self.memory) >= self.batch_size:
                self.train_using_memory()

            # print(f'Episode: {e:4}/{self.NUM_EPISODES} and step: {t:4}. Eps: {float(self.ETA):.2}, reward {reward}')

            eta_list.append(self.ETA)
            reward_list.append(fin_reward)

            if self.VERVOSE: print(f"--> Reward: {fin_reward}, #Steps: {num_steps}, Eta: {float(self.ETA):.2} \
                        Avg Rewards: {(np.average(fin_reward))}, steps: {num_steps}")                        
        print(' Train mean % score= ', round(100*np.mean(reward_list),1))

        # Env close
        self.env.close()

        return reward_list