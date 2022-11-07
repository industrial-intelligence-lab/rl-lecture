import numpy as np
from collections import defaultdict
from Omok_Utils import Omok_Utils

class OmokPlayer_MC_Agent(object):

    # game id
    GAME_ID = "내꺼MC1호기"

    # Global vars
    Q = {}                             # {s: [a,...,a]}
    Returns = defaultdict(list)        # {(s, a): [r,...,r]}
    traj = []
    last_s, last_a = -1, -1
    tot_rewards = []

    ut = Omok_Utils()

    def __init__(self, board_size, ETA=0.2, GAMMA=0.9, VERVOSE=True, REPORTING=True):
        self.board_size = board_size
        self.ETA = ETA
        self.GAMMA = GAMMA
        self.VERVOSE = VERVOSE
        self.REPORTING = REPORTING
        self.Q = defaultdict(lambda: np.random.rand(board_size**2))

    def episode_start(self):
        self.traj = []

    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values==values.max()))

    def action_tuple_to_num(self, tuple):
        return tuple[0]*9 + tuple[1]

    def action_num_to_tuple(self, num):
        return (num // 9, num % 9)

    # get action called from Omok_Env
    def get_action(self, board, p_id):

        # 마지막 s, a와 최신의 r을 기록
        if self.last_s != -1: self.traj.append((self.last_s, self.last_a, 0.0))

        a = -1
        s = self.ut.hash(board, p_id)
        if np.random.rand() < self.ETA:
            a = np.random.randint(self.board_size**2)
            if self.VERVOSE: print('Random action %d for %s -> %s (%s)' % (s, p_id, self.action_num_to_tuple(a), a))
        else:
            # Find a*
            a = self.random_argmax(self.Q[s])
            if self.VERVOSE: print('Greedy action %d for %s -> %s (%s)' % (s, p_id, self.action_num_to_tuple(a), a))

        self.last_s = s
        self.last_a = a

        return self.action_num_to_tuple(a)

    def episode_end(self, winner_idx_idx, winner_idx_id, board, p_id, history):
        
        # record the last transition
        s = self.ut.hash(board, p_id)
        r = 1.0 if winner_idx_id == self.get_id() else 0.0
        self.traj.append((self.last_s, self.last_a, r))
        self.tot_rewards.append(r)

        # G
        G = 0

        # For each step
        for i, (s, a, r) in reversed(list(enumerate(self.traj))):
            # Update G
            G = self.GAMMA*G + r
            first_idx = next(j for j, t in enumerate(self.traj) if t[0] == s and t[1] == a)
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

    def get_id(self):
        return self.GAME_ID 


