class OmokPlayer(object):

    def get_action(self, board, p_id):
        # print state
        # for x in board.states_loc:
        #     print(x)
        # print(board.current_state_for_me())

        print("돌을 둘 좌표를 입력하세요 %s(%d)" % (self.get_id(), p_id))
        location = input()
        if isinstance(location, str) : location = [int(n, 10) for n in location.split(",")]        
        return location

    def get_id(self):
        return "데지나라"