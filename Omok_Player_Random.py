import numpy as np

class OmokPlayer_Random(object):

    # game id
    GAME_ID = "랜덤플레이어"

    def __init__(self, board_size):
        self.board_size = board_size

    def episode_start(self):
        pass

    def get_action(self, board, p_id):

        a = (np.random.randint(0, self.board_size), (np.random.randint(0, self.board_size)))
        print("좌표선택", a, self.get_id(), p_id)                        
        return a        

    def episode_end(self, winer_idx, winder_id, history):
        pass

    def get_id(self):
        return self.GAME_ID