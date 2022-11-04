from Omok_Utils import Omok_Utils

class OmokPlayer(object):

    ut = Omok_Utils()

    def get_action(self, board, p_id):
                
        s = self.ut.hash(board, p_id)
        print(s)

        print("돌을 둘 좌표를 입력하세요 %s(%d)" % (self.get_id(), p_id))
        location = input()
        if isinstance(location, str) : location = [int(n, 10) for n in location.split(",")]        
        return location

    def get_id(self):
        return "데지나라"