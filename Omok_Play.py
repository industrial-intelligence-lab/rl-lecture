import random
from Omok_Env import Board, Game
from Omok_Player_Random import OmokPlayer_Random as h1
# from Omok_Player_Random import OmokPlayer_Random as h2
from Omok_Player_Human import OmokPlayer as h2

if __name__ == '__main__':
    width, height = 9, 9
    n = 3
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)
    human1 = h1()
    human2 = h2()

    game.SLEEP_TIME = 0
    
    winer, history = game.start_play(human1, human2, start_player=random.randint(1,2), is_shown=1)
    # print(winer)
    for h in history:
        print(h)

