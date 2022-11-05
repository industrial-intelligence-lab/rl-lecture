import random
import time
from Omok_Env import Board, Game
from Omok_Player_Random import OmokPlayer_Random as a1
# from Omok_Player_MC import OmokPlayer_MC_Agent as a1
from Omok_Player_MC import OmokPlayer_MC_Agent as a2
# from Omok_Player_Human import OmokPlayer as h1
# from Omok_Player_Human import OmokPlayer as h2

IS_SHOWN = 0
IS_SHOWN_AFTER = 10000

if __name__ == '__main__':
    board_size = 9
    n = 3
    board = Board(width=board_size, height=board_size, n_in_row=n)
    game = Game(board)
    agent1 = a1(board_size) #, ETA=0.2, GAMMA=0.9, VERVOSE=False, REPORTING=False)
    agent2 = a2(board_size, ETA=0.2, GAMMA=0.9, VERVOSE=False, REPORTING=False)
    agent2.GAME_ID = '2호기'

    game.SLEEP_TIME = 1
    
    win_cnt = [0, 0]

    for e in range(500000):
        print(e, '-'*10)
        if IS_SHOWN_AFTER > 0 and e > IS_SHOWN_AFTER:
            IS_SHOWN = 1
        winner_idx, winner_id, history = \
            game.start_play(agent1, agent2, start_player=random.randint(1,2), is_shown=IS_SHOWN)                
        win_cnt[winner_idx-1] += 1
        print(winner_idx, winner_id, 
            "{:.2f}".format(win_cnt[0]/sum(win_cnt)),"{:.2f}".format(win_cnt[1]/sum(win_cnt)))
        # for h in history:
        #     print(h)
        if IS_SHOWN == 1: time.sleep(5)

