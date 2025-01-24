import numpy as np


class TTTNvN:
    def __init__(
        self,
        nfirst=2,
        n_moves=2,
        render_mode="human",
        random_op=True,
        obs_as_array=True,
    ):
        self.obs_as_array = obs_as_array
        self.n_moves = n_moves
        self.nfirst = nfirst
        self.board = 0
        self.random_op = random_op
        self.needs_to_reset = True

        self.current_player = 0
        self.render_mode = render_mode == "human"
        self.wc = np.array(
            [
                2**0 + 2**1 + 2**2,  # row 1
                2**3 + 2**4 + 2**5,  # row 2
                2**6 + 2**7 + 2**8,  # row 3
                2**0 + 2**3 + 2**6,  # col 1
                2**1 + 2**4 + 2**7,  # col 2
                2**2 + 2**5 + 2**8,  # col 3
                2**0 + 2**4 + 2**8,  # diag 1
                2**2 + 2**4 + 2**6,  # diag 2
            ]
        )
        self.wc2 = np.left_shift(self.wc, 9)
        self.b1 = np.sum(
            np.left_shift(np.ones(9, dtype=np.int32), np.arange(0, 9, dtype=np.int32))
        )
        self.b2 = np.sum(
            np.left_shift(np.ones(9, dtype=np.int32), np.arange(9, 18, dtype=np.int32))
        )

    def random_legal_moves(self, n=1, board=0):
        x = (
            np.bitwise_and(
                board,
                np.left_shift(
                    np.ones(9, dtype=np.int32), np.arange(0, 9, dtype=np.int32)
                ),
            )
            > 0
        )
        y = (
            np.bitwise_and(
                board,
                np.left_shift(
                    np.ones(9, dtype=np.int32), np.arange(9, 18, dtype=np.int32)
                ),
            )
            > 0
        )
        legal = np.logical_not(np.logical_or(x, y))
        c = np.random.choice(
            a=np.arange(0, 9), p=legal / np.sum(legal), size=n, replace=True
        )
        return c

    def make_move(self, i, board):
        board |= ((~board & (1 << (i + (1 - self.current_player) * 9))) > 0) * (
            1 << (i + self.current_player * 9)
        )
        return board

    def check_win(self, board, player):
        if player == 0:
            return np.sum(np.bitwise_and(board, self.wc) == self.wc) > 0
        else:
            return np.sum(np.bitwise_and(board, self.wc2) == self.wc2) > 0

    def check_draw(self, board):
        return (board & self.b1) | ((board & self.b2) >> 9) == 511

    def reset(self):
        self.needs_to_reset = False
        self.board = 0
        self.current_player = 0
        return self.board_to_array(0) if self.obs_as_array else 0, 0

    def display_board(self, board: int):
        board_str = ""
        for y in range(3):
            for x in range(3):
                b = f"{'x' if board & (1<<(x+3*y)) else ' '}{'y' if board & (1<<(x+3*y+9)) else ' '},"
                board_str = board_str + b
            board_str += "\n"
        print(board_str)

    def board_to_array(self, board):
        return (
            np.bitwise_and(
                np.left_shift(
                    np.ones(18, dtype=np.int32), np.arange(18, dtype=np.int32)
                ),
                board,
            )
            > 0
        ).astype(np.float32)

    def step(self, action=0):
        win = 0
        op_win = 0
        if self.needs_to_reset:
            print("WARNING ENV NEEDS TO RESET")
            return 0, 0, 0, 0, 0

        if not isinstance(action, (list, tuple, np.ndarray)):
            action = [action]
        for i in range(len(action)):
            self.board = self.make_move(action[i], self.board)

        if self.render_mode:
            self.display_board(self.board)
        win = self.check_win(self.board, self.current_player)
        draw = False
        if not win:
            draw = self.check_draw(self.board)
        done = win or draw
        self.current_player = 1 - self.current_player

        if self.random_op and not done:
            m = self.random_legal_moves(self.n_moves, self.board)
            if not isinstance(m, (list, tuple, np.ndarray)):
                m = [m]
            for i in range(self.n_moves):
                self.board = self.make_move(m[i], self.board)
            op_win = self.check_win(self.board, self.current_player)
            if not op_win:
                draw = self.check_draw(self.board)
            self.current_player = 1 - self.current_player
            done = op_win or draw

        if done:
            self.needs_to_reset = True

        obs = self.board
        if self.obs_as_array:
            obs = self.board_to_array(self.board)
        return (
            obs,
            float(win) - float(op_win),
            done,
            False,
            0,
        )  # obs, r term, trunc, info


class TTTLeverGame:
    def __init__(
        self,
        n_moves=2,
        render_mode="human",
        random_op=True,
        obs_as_array=True,
    ):
        self.obs_as_array = obs_as_array
        self.n_moves = n_moves
        self.nfirst = n_moves
        self.board = 0
        self.random_op = random_op
        self.needs_to_reset = True

        self.current_player = 0
        self.render_mode = render_mode == "human"
        self.wc = np.array(
            [
                2**0 + 2**1 + 2**2,  # row 1
                2**3 + 2**4 + 2**5,  # row 2
                2**6 + 2**7 + 2**8,  # row 3
                2**0 + 2**3 + 2**6,  # col 1
                2**1 + 2**4 + 2**7,  # col 2
                2**2 + 2**5 + 2**8,  # col 3
                2**0 + 2**4 + 2**8,  # diag 1
                2**2 + 2**4 + 2**6,  # diag 2
            ]
        )
        self.wc2 = np.left_shift(self.wc, 9)
        self.b1 = np.sum(
            np.left_shift(np.ones(9, dtype=np.int32), np.arange(0, 9, dtype=np.int32))
        )
        self.b2 = np.sum(
            np.left_shift(np.ones(9, dtype=np.int32), np.arange(9, 18, dtype=np.int32))
        )

    def random_legal_move(self, board=0):
        x = (
            np.bitwise_and(
                board,
                np.left_shift(
                    np.ones(9, dtype=np.int32), np.arange(0, 9, dtype=np.int32)
                ),
            )
            > 0
        )
        y = (
            np.bitwise_and(
                board,
                np.left_shift(
                    np.ones(9, dtype=np.int32), np.arange(9, 18, dtype=np.int32)
                ),
            )
            > 0
        )
        legal = np.logical_not(np.logical_or(x, y))
        c = np.random.choice(
            a=np.arange(0, 9), p=legal / np.sum(legal), size=1, replace=True
        )
        return c

    def make_move(self, i, board):
        board |= ((~board & (1 << (i + (1 - self.current_player) * 9))) > 0) * (
            1 << (i + self.current_player * 9)
        )
        return board

    def check_win(self, board, player):
        if player == 0:
            return np.sum(np.bitwise_and(board, self.wc) == self.wc) > 0
        else:
            return np.sum(np.bitwise_and(board, self.wc2) == self.wc2) > 0

    def check_draw(self, board):
        return (board & self.b1) | ((board & self.b2) >> 9) == 511

    def reset(self):
        self.needs_to_reset = False
        self.board = 0
        self.current_player = 0
        return self.board_to_array(0) if self.obs_as_array else 0, 0

    def display_board(self, board: int):
        board_str = ""
        for y in range(3):
            for x in range(3):
                b = f"{'x' if board & (1<<(x+3*y)) else ' '}{'y' if board & (1<<(x+3*y+9)) else ' '},"
                board_str = board_str + b
            board_str += "\n"
        print(board_str)

    def board_to_array(self, board):
        return (
            np.bitwise_and(
                np.left_shift(
                    np.ones(18, dtype=np.int32), np.arange(18, dtype=np.int32)
                ),
                board,
            )
            > 0
        ).astype(np.float32)

    def step(self, actions=0):
        win = 0
        op_win = 0
        if self.needs_to_reset:
            print("WARNING ENV NEEDS TO RESET")
            return 0, 0, 0, 0, 0

        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions] * self.n_moves
        ac0 = actions[0]
        matching = True
        for ac in actions:  # TODO: replace with np.sum(np.equals...) to see if faster
            if ac0 != ac:
                matching = False

        if matching:
            self.board = self.make_move(actions[0], self.board)

        if self.render_mode:
            self.display_board(self.board)
        win = self.check_win(self.board, self.current_player)
        draw = False
        if not win:
            draw = self.check_draw(self.board)
        done = win or draw
        self.current_player = 1 - self.current_player

        if self.random_op and not done:
            m = self.random_legal_move(self.board)
            self.board = self.make_move(m, self.board)
            op_win = self.check_win(self.board, self.current_player)
            if not op_win:
                draw = self.check_draw(self.board)
            self.current_player = 1 - self.current_player
            done = op_win or draw

        if done:
            self.needs_to_reset = True

        obs = self.board
        if self.obs_as_array:
            obs = self.board_to_array(self.board)
        return (
            obs,
            float(win) - float(op_win),
            done,
            False,
            0,
        )  # obs, r term, trunc, info


if __name__ == "__main__":
    n_actions = 3
    ttenv = TTTLeverGame(n_moves=n_actions, render_mode="human", random_op=True)
    ttenv.reset()
    ttenv.current_player = 0

    again = "y"
    while again == "y":
        # ttenv.board2 = np.random.randint(0, 2**18 + 1, dtype=np.int32)
        na = []
        for i in range(n_actions):
            na.append(int(input("action 1-9: ")))
        m = ttenv.random_legal_move(ttenv.board)
        obs, r, term, trunc, _ = ttenv.step(na)
        ttenv.display_board(ttenv.board)
        print(
            f"obs: {obs}, reward: {r}, terminated: {term}, truncated: {trunc}, info: {_}, act: {m}"
        )
        if ttenv.needs_to_reset:
            ttenv.reset()
        again = input("try again? ")
    ttenv.display_board(0)
    ttenv.display_board(1)
    ttenv.display_board(2)
    ttenv.display_board(4)
    ttenv.display_board(8)
    ttenv.display_board(16)
    ttenv.display_board(32)
    ttenv.display_board(64)
    ttenv.display_board(128)
    ttenv.display_board(256)
    ttenv.display_board(512)
    ttenv.display_board(257)

    ttenv = TTTNvN(1, render_mode="human")
    done = False
    obs, info = ttenv.reset()
    nepisodes = 5
    for i in range(nepisodes):
        print(f"Starting episode: {i}")
        done = False
        obs, info = ttenv.reset()
        while not done:
            act = np.random.randint(0, 9)
            obs, reward, terminated, truncated, info = ttenv.step(act)
            print(
                f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}, act: {act}"
            )
            done = terminated or truncated
            input()
