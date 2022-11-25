import random
import time
from Omok_Env import Board, Game
import matplotlib.pyplot as plt
import numpy as np

from Omok_Player_Random import OmokPlayer_Random as a1
# from Omok_Player_MC import OmokPlayer_MC_Agent as a1
# from Omok_Player_MC import OmokPlayer_MC_Agent as a2
# from Omok_Player_Human import OmokPlayer as h1
# from Omok_Player_Human import OmokPlayer as h2
from Omok_Player_Sarsa import OmokPlayer_Sarsa_Agent as sarsa
from Omok_Player_Qlearning import OmokPlayer_Qlearning_Agent as q

IS_SHOWN = 0
IS_SHOWN_AFTER = 5000000
REPORTING = False

tot_rewards = [[],[]]

# performance graph
def perf_graph(i):    
    plt.cla()
    print(tot_rewards)
    plt.plot(range(len(np.array(tot_rewards[0], dtype=int))), tot_rewards[0], label='a1')
    plt.plot(range(len(np.array(tot_rewards[1], dtype=int))), tot_rewards[1], label='a2')

if __name__ == '__main__':
    board_size = 5
    n = 3
    board = Board(width=board_size, height=board_size, n_in_row=n)
    game = Game(board)
    agent1 = a1(board_size) #, ETA=0.2, GAMMA=0.9, VERVOSE=False, REPORTING=False)
    agent2 = q(board_size, ETA=0.05, GAMMA=0.9, ALPHA=0.15, VERVOSE=False, REPORTING=False)
    # agent2.GAME_ID = '2호기'

    game.SLEEP_TIME = 1
    
    win_cnt = [0, 0]

    for e in range(100000000):
        
        if IS_SHOWN_AFTER > 0 and e > IS_SHOWN_AFTER:
            IS_SHOWN = 1
        winner_idx, winner_id, history = \
            game.start_play(agent1, agent2, start_player=random.randint(1,2), is_shown=IS_SHOWN)                
        if winner_idx != -1: win_cnt[winner_idx-1] += 1
        if e%10000==0 or IS_SHOWN: 
            print(e, '-'*10)
            print(winner_idx, winner_id, 
                "{:.3f}".format(win_cnt[0]/sum(win_cnt)),"{:.3f}".format(win_cnt[1]/sum(win_cnt)))
        if e%10000==0: win_cnt = [0, 0]
        # for h in history:
        #     print(h)
        if IS_SHOWN == 1: time.sleep(5)
        
        if winner_idx == 1: 
            tot_rewards[0].append(1)
            tot_rewards[1].append(0)
        elif winner_idx == 2: 
            tot_rewards[0].append(0)
            tot_rewards[1].append(1)
        else: 
            tot_rewards[0].append(0) 
            tot_rewards[1].append(0)

        if REPORTING:
            plt.pause(0.001)
            perf_graph(0)


    if REPORTING:                        
        plt.tight_layout()
        plt.show()