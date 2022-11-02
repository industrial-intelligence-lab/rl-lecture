class Human(object):

    def get_action(self, board):
        try:
            print("돌을 둘 좌표를 입력하세요", self.get_id())
            location = input()
            if isinstance(location, str) : location = [int(n, 10) for n in location.split(",")]        
            move = board.location_to_move(location)
        except Exception as e : move = -1
            
        if move == -1 or move in board.states.keys() :
            print("다시 입력하세요.")
            move = self.get_action(board)            
        return move

    def get_id(self):
        return "내꺼1"
