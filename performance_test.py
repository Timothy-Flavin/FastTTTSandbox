import numpy as np


class TTT2v2:
    def __init__(self, nfirst=2):
        self.nfirst = nfirst
        self.board = np.zeros(18, dtype=np.int32)
        self.board2 = 0
        self.current_player = 0

    def make_move1(self, i):
        self.board2 += (self.board2 & (1 << (i * ((1 - self.current_player) + 1)))) * (
            1 << (i * self.current_player + 1)
        )

    def display_board2(self, board):
        board_str = ""
        for y in range(3):
            for x in range(3):
                b = f"   ,"
        continue
