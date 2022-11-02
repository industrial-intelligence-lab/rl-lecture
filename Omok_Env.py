import numpy as np
# from Omok_Renju_Rule import Renju_Rule # 금수없음
from IPython.display import clear_output
import os

class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.n_in_row = int(kwargs.get('n_in_row', 5))

    def init_board(self, start_player=0) :
        self.current_player = start_player 
        self.last_move, self.last_loc = -1, -1
        
        self.states, self.states_loc = {}, [[0] * self.width for _ in range(self.height)]
        self.forbidden_locations, self.forbidden_moves = [], []
      
    def move_to_location(self, move):
        """ 3*3 보드를 예로 들면 : move 5 는 좌표 (1,2)를 의미한다.""" # ex) 0 1 2
        h = move // self.width                                      #     3 4 5
        w = move % self.width                                       #     6 7 8
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2 : return -1
        h, w = location[0], location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height) : return -1
        return move

    def current_state(self):
        """현재 플레이어의 관점에서 보드 상태(state)를 return한다.
        state shape: 4 * [width*height] """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0 #내가 둔 돌의 위치를 1로 표현
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0 #적이 둔 돌의 위치를 1로 표현
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0 #마지막 돌의 위치
            
        if len(self.states) % 2 == 0 : square_state[3][:, :] = 1.0  # indicate the colour to play
        
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        loc = self.move_to_location(move)
        self.states_loc[loc[0]][loc[1]] = self.current_player + 1 #1 if self.is_you_black() else 2
        # self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.current_player = (0 if self.current_player == 1 else 1)
        self.last_move, self.last_loc = move, loc

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        # moved : 이미 돌이 놓인 자리들
        moved = list(self.states.keys())        
        if len(moved) < self.n_in_row * 2-1 : return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player
        
        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()        
        print(win, winner)
        if win : return True, winner        
        elif len(self.states) == self.width*self.height : return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Game(object):
    players = {}

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, p1_idx = 0, p2_idx = 1):
        width = board.width
        height = board.height

        p1_id = self.players[p1_idx].get_id()
        p2_id = self.players[p2_idx].get_id()
        cur_p_id = self.players[board.current_player].get_id()

        clear_output(wait=True)
        os.system('cls')
        
        print()
        print("흑돌(●) : 플레이어", p1_id)
        print("백돌(○) : 플레이어", p2_id)
        print("--------------------------------\n")
                
        print(cur_p_id, "당신의 차례입니다.\n")
            
        row_number = ['⒪','⑴','⑵','⑶','⑷','⑸','⑹','⑺','⑻','⑼','⑽','⑾','⑿','⒀','⒁']
        # row_number = range(14)
        print('　', end='')
        for i in range(height) : print(row_number[i], end='')
        print()
        for i in range(height):
            print(row_number[i], end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == p1_idx : print('●', end='')
                elif p == p2_idx : print('○', end='')                
                else : print('　', end='')
            print()
        if board.last_loc != -1 :
            print(f"플레이어 {cur_p_id}의 수 : ({board.last_loc[0]},{board.last_loc[1]})\n")

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        self.board.init_board(start_player)
        self.players = {0: player1, 1: player2}
        while True:
            if is_shown : self.graphic(self.board)
                
            current_player = self.board.get_current_player()
            player_in_turn = self.players[current_player]

            move = player_in_turn.get_action(self.board, current_player+1)
                
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    self.graphic(self.board)
                    if winner != -1 : print("Game end. Winner is", self.players[winner].get_id())
                    else : print("Game end. Tie")
                return winner
            

    
    '''
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ 스스로 자가 대국하여 학습 데이터(state, mcts_probs, z) 생성 """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            # 흑돌일 때, 금수 위치 확인하기
            # if self.board.is_you_black() : self.board.set_forbidden()
            if is_shown : self.graphic(self.board, p1, p2)
                
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            
            # perform a move
            self.board.do_move(move)
                
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    self.graphic(self.board, p1, p2)
                    if winner != -1 : print("Game end. Winner is player:", winner)
                    else : print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
    '''