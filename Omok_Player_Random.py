import numpy as np

class OmokPlayer_Random(object):

    def get_action(self, board, p_id):

        a = (np.random.randint(0, 9), (np.random.randint(0, 9)))
        print("좌표선택", a, self.get_id(), p_id)                        
        return a        

    def get_id(self):
        return "랜덤플레이어"
