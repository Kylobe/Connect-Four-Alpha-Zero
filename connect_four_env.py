from connect_four import Board
import numpy as np
import torch



class ConnectFourEnv:
    def __init__(self):
        self.board = Board()
        self.num_moves = 7

    def reset(self):
        self.board = Board()
        return self.board

    def step(self, action):
        if action not in self.board.legal_moves:
            print("Illegal Move Chosen")
            return self.board.copy(), True, None
        self.board.push_move(action)
        winner, done = self.board.get_result()
        return self.board.copy(), done, winner

    @staticmethod
    def create_action_mask(board: Board, device=None):
        mask = torch.zeros(7, dtype=torch.float32, device=device)
        for move in board.legal_moves:
            mask[move] = 1.0
        return mask

    @staticmethod
    def encode_board(board: Board):
        if not board.current_turn:
            board = board.get_inverse_board()
        piece_planes = torch.zeros((2, 6, 7), dtype=torch.float32)
        for row in range(6):
            for col in range(7):
                cur_cell = board.get_square(row, col)
                if not cur_cell is None:
                    cur_plane_idx = 0 if cur_cell else 1
                    piece_planes[cur_plane_idx][row][col] = 1.0
        return piece_planes

