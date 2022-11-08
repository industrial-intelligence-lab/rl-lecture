import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Sarsa_On_policy:

    # Global vars
    Q = {}                             # {s: [a,...,a]}
    Returns = defaultdict(list)        # {(s, a): [r,...,r]}

    def __init__(self, env, NUM_ACTS, NUM_EPISODES=100, ETA=0.1, GAMMA=0.9, ALPHA=0.1, VERVOSE=True, REPORTING=True):
        self.env = env
        self.NUM_ACTS = NUM_ACTS
        self.NUM_EPISODES = NUM_EPISODES
        self.ETA = ETA
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.VERVOSE = VERVOSE
        self.REPORTING = REPORTING
        self.Q = defaultdict(lambda: np.random.rand(NUM_ACTS))

    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values==values.max()))

    # get action with e-greedy
    def get_action(self, s):
        a = -1
        if np.random.rand() < self.ETA:
            a = np.random.randint(self.NUM_ACTS)
            if self.VERVOSE: print('Random action for %s -> %s' % (s, a))
        else:
            # Find a*
            a = self.random_argmax(self.Q[s])
            if self.VERVOSE: print('Greedy action for %s -> %s' % (s, a))
        return a

    # 전체 episode와 time step을 반복하면서 학습진행->종료
    def do_train_e_t(self):
        # Reporting
        tot_rewards = []
        tot_steps = []

        for e in range(self.NUM_EPISODES):
            if self.VERVOSE: print("\rEpisode {}/{} --------------------- ".format(e, self.NUM_EPISODES), end="\n")

            # Env reset
            s, info = self.env.reset()

            # Terminiation condition
            terminated = False
            truncated = False

            num_steps = 0
            fin_reward = 0
            # Generate an episode
            while not terminated and not truncated: 
                # a = np.random.choice(NUM_ACTS, p=PI[s])
                a = self.get_action(s)
                next_s, r, terminated, truncated, info = self.env.step(a)   
                # Sarsa update
                next_a = self.get_action(next_s)
                # set Q for terminated or truncated
                if terminated:
                    self.Q[next_s][next_a] = 0.0                
                td_target = r + self.GAMMA * self.Q[next_s][next_a]
                td_error = td_target - self.Q[s][a]
                old_Q = self.Q[s][a]
                self.Q[s][a] += self.ALPHA * td_error

                if self.VERVOSE: print("State:", s, "Action", a, "Reward:", r, "Next state:", next_s, "Next action:", next_a, \
                                                     "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
                # if self.VERVOSE: print('Q', old_Q, '->', self.Q[s][a], 'td_target:',td_target, 'td_error:', td_error)
                
                s = next_s
                num_steps += 1
                fin_reward += r
            tot_rewards.append(fin_reward)
            tot_steps.append(num_steps)

            if self.VERVOSE: print(f"--> Reward: {fin_reward}, #Steps: {num_steps}, \
                        Avg Rewards: {(np.average(tot_rewards))}, Avg steps: {np.average(tot_steps)}")                        

        # Env close
        self.env.close()

        # performance graph
        def perf_graph(i):    
            plt.cla()
            print()
            plt.plot(range(len(np.array(tot_rewards, dtype=int))), tot_rewards)

        if self.REPORTING:
            perf_graph(1)
            plt.tight_layout()
            plt.show()    

        return tot_rewards