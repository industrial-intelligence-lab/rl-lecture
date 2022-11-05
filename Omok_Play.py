import random
import time
from Omok_Env import Board, Game
from Omok_Player_Random import OmokPlayer_Random as a1
from Omok_Player_MC import OmokPlayer_MC_Agent as a2
# from Omok_Player_Human import OmokPlayer as h1
# from Omok_Player_Human import OmokPlayer as h2

if __name__ == '__main__':
    board_size = 9
    n = 3
    board = Board(width=board_size, height=board_size, n_in_row=n)
    game = Game(board)
    agent1 = a1(board_size)
    agent2 = a2(board_size)

    game.SLEEP_TIME = 1
    

    for e in range(3):
        print(e, '-'*10)
        agent1.episode_start()
        agent2.episode_start()
        winer_idx, winder_id, history = game.start_play(agent1, agent2, start_player=random.randint(1,2), is_shown=1)                
        agent1.episode_end(winer_idx, winder_id, history)
        agent2.episode_end(winer_idx, winder_id, history)
        print(winer_idx, winder_id)
        for h in history:
            print(h)
        time.sleep(5)

