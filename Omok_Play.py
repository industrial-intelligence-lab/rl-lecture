from Omok_Env import Board, Game
from Omok_Player1 import Human as h1
from Omok_Player2 import Human as h2

if __name__ == '__main__':
    width, height = 15, 15
    n = 2
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)
    human1 = h1()
    human2 = h2()
    game.start_play(human1, human2, start_player=1, is_shown=1)
