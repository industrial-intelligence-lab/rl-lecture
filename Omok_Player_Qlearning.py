import numpy as np
from collections import defaultdict
from Omok_Utils import Omok_Utils

class OmokPlayer_Qlearning_Agent(object):

    # game id
    GAME_ID = "내꺼Q1호기"

    # Global vars
    Q = {}                             # {s: [a,...,a]}
    Returns = defaultdict(list)        # {(s, a): [r,...,r]}
    last_s, last_a = -1, -1
    tot_rewards = []

    ut = Omok_Utils()

    def __init__(self, board_size, ETA=0.2, GAMMA=0.9, ALPHA=0.1, VERVOSE=True, REPORTING=True):
        self.board_size = board_size
        self.ETA = ETA
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.VERVOSE = VERVOSE
        self.REPORTING = REPORTING
        self.Q = defaultdict(lambda: np.random.rand(board_size**2))

    def episode_start(self):
        self.last_s, self.last_a = -1, -1

    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values==values.max()))

    def action_tuple_to_num(self, tuple):
        return tuple[0]*9 + tuple[1]

    def action_num_to_tuple(self, num):
        return (num // 9, num % 9)

    # get action with e-greedy
    def get_action_policy(self, s):
        a = -1
        if np.random.rand() < self.ETA:
            a = np.random.randint(self.board_size**2)
            if self.VERVOSE: print('Random action %d for %s -> %s (%s)' % (s, p_id, self.action_num_to_tuple(a), a))
        else:
            # Find a*
            a = self.random_argmax(self.Q[s])
            if self.VERVOSE: print('Greedy action %d for %s -> %s (%s)' % (s, p_id, self.action_num_to_tuple(a), a))
        return a

    # get action called from Omok_Env
    def get_action(self, board, p_id):

        s = self.ut.hash(board, p_id)
        a = self.get_action_policy(s)

        # q training (self.last_s, self.last_a, r=0, s)
        if self.last_s != -1: 
            r = 0
            next_best_a = self.random_argmax(self.Q[s])
            td_target = r + self.GAMMA * self.Q[s][next_best_a]
            td_error = td_target - self.Q[self.last_s][self.last_a]
            self.Q[self.last_s][self.last_a] += self.ALPHA * td_error

            # if self.VERVOSE: print("State:", s, "Action", a, "Reward:", r, "Next state:", next_s, "Next action:", next_a, \
            #                                         "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
            # if self.VERVOSE: print('Q', old_Q, '->', self.Q[s][a], 'td_target:',td_target, 'td_error:', td_error)

        self.last_s = s
        self.last_a = a

        return self.action_num_to_tuple(a)

    def episode_end(self, winner_idx_idx, winner_idx_id, board, p_id, history):
        
        # record the last transition
        # s = self.ut.hash(board, p_id)
        r = 1.0 if winner_idx_id == self.get_id() else 0.0
        self.tot_rewards.append(r)

        # q training for the last (self.last_s, self.last_a, r, s)
        td_error = r - self.Q[self.last_s][self.last_a]
        self.Q[self.last_s][self.last_a] += self.ALPHA * td_error

    def get_id(self):
        return self.GAME_ID 


