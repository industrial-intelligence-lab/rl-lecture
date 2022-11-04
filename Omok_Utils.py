import numpy as np 
from pprint import pprint

class Omok_Utils:

    def change_7_for_me_9_for_opponent(self, board, p_id):
        # print('board',board)
        x = np.array(board.states_loc)
        x[x == p_id] = 7
        x[x == 1] = 9
        x[x == 2] = 9
        return x

    def hash(self, board, p_id):
        x = self.change_7_for_me_9_for_opponent(board, p_id)
        b = tuple(map(tuple, x))
        # pprint(b)
        return hash(b)