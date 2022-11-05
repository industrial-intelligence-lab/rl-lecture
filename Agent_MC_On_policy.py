import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MC_On_policy:

    # Global vars
    Q = {}                             # {s: [a,...,a]}
    Returns = defaultdict(list)        # {(s, a): [r,...,r]}

    # Reporting
    tot_rewards = []
    tot_steps = []

    def __init__(self, env, NUM_ACTS, NUM_EPISODES=100, ETA=0.2, GAMMA=0.9, VERVOSE=True, REPORTING=True):
        self.env = env
        self.NUM_ACTS = NUM_ACTS
        self.NUM_EPISODES = NUM_EPISODES
        self.ETA = ETA
        self.GAMMA = GAMMA
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
        for e in range(self.NUM_EPISODES):
            print("\rEpisode {}/{} --------------------- ".format(e, self.NUM_EPISODES), end="\n")

            # Env reset
            s, info = self.env.reset()

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
                a = self.get_action(s)
                next_s, r, terminated, truncated, info = self.env.step(a)       
                traj.append((s, a, r))
                if self.VERVOSE: print("State:", s, "Action", a, "Reward:", r, "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
                s = next_s
                num_steps += 1
                fin_reward += r
            if self.REPORTING: self.tot_rewards.append(fin_reward)
            if self.REPORTING: self.tot_steps.append(num_steps)

            print(f"--> Reward: {fin_reward}, #Steps: {num_steps}, \
                        Avg Rewards: {(np.average(self.tot_rewards))}, Avg steps: {np.average(self.tot_steps)}")

            # Check trajectory
            if self.VERVOSE:
                for i, t in enumerate(traj):
                    print(i, t)

            # G
            G = 0

            # For each step
            for i, (s, a, r) in reversed(list(enumerate(traj))):
                # Update G
                G = self.GAMMA*G + r
                first_idx = next(j for j, t in enumerate(traj) if t[0] == s and t[1] == a)
                if self.VERVOSE: print(i, (s, a, r), first_idx)
                if i == first_idx:
                    # print(i, (s, a, r))
                    # Append G to (s, a)
                    self.Returns[(s, a)].append(G)
                    # Check returns
                    if self.VERVOSE: print('Returns(%s,%s) -> %s' % (s, a, self.Returns[(s, a)]))            
                    # Update Q
                    self.Q[s][a] = sum(self.Returns[(s, a)]) / len(self.Returns[(s, a)])
                    if self.VERVOSE: print('Update Q(%s,%s) -> %f' % (s, a, self.Q[s][a]))

        # Env close
        self.env.close()

        # performance graph
        def perf_graph(i):    
            plt.cla()
            print()
            plt.plot(range(len(np.array(self.tot_rewards, dtype=int))), self.tot_rewards)

        if self.REPORTING:
            perf_graph(1)
            plt.tight_layout()
            plt.show()    
