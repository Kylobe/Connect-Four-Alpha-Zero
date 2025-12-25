
class Board:
    def __init__(self):
        self.board = [[None for _ in range(7)] for _ in range(6)]
        self.current_turn = True
        self.game_over = False
        self.winner = None
        self.top_row_count = 0

    def copy(self):
        new_board = [[self.board[row][col] for col in range(7)] for row in range(6)]
        new_board_obj = Board()
        new_board_obj.board = new_board
        new_board_obj.current_turn = self.current_turn
        new_board_obj.game_over = self.game_over
        new_board_obj.winner = self.winner
        new_board_obj.top_row_count = self.top_row_count
        return new_board_obj

    def get_next_state(self, column: int): 
        new_board = self.copy()
        board = new_board.board
        placed_piece = False
        winner = None
        done = False
        for row in range(6):
            if not board[row][column] is None:
                if row == 0:
                    raise ValueError("Illegal Move")
                board[row - 1][column] = self.current_turn
                winner, done = new_board.check_winning_move(row - 1, column)
                placed_piece = True
                if row - 1 == 0:
                    new_board.top_row_count += 1
                    if new_board.top_row_count >= 7:
                        done = True
                break
        if not placed_piece:
            board[5][column] = self.current_turn
            winner, done = new_board.check_winning_move(5, column)
        new_board.winner = winner
        new_board.game_over = done
        new_board.current_turn = not new_board.current_turn
        return new_board

    def check_winning_move(self, row, col):
        if not self.board[row][col] is None:
            cur_cell = self.board[row][col]
            if row < 3:
                count = 1
                for cur_row in range(row + 1, row + 4):
                    if self.board[cur_row][col] == cur_cell:
                        count += 1
                    else:
                        break
                if count >= 4:
                    return self.board[row][col], True
            count = 1
            for cur_col in range(col + 1, min(col + 4, 7)):
                if self.board[row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            for cur_col in range(col - 1, max(col - 4, -1), -1):
                if self.board[row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            if count >= 4:
                return self.board[row][col], True
            count = 1
            for cur_row, cur_col in zip(range(row + 1, min(row + 4, 6)), range(col + 1, min(col + 4, 7))):
                if self.board[cur_row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            for cur_row, cur_col in zip(range(row - 1, max(row - 4, -1), -1), range(col - 1, max(col - 4, -1), -1)):
                if self.board[cur_row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            if count >= 4:
                return self.board[row][col], True
            count = 1
            for cur_row, cur_col in zip(range(row + 1, min(row + 4, 6)), range(col - 1, max(col - 4, -1), -1)):
                if self.board[cur_row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            for cur_row, cur_col in zip(range(row - 1, max(row - 4, -1), -1), range(col + 1, min(col + 4, 7))):
                if self.board[cur_row][cur_col] == cur_cell:
                    count += 1
                else:
                    break
            if count >= 4:
                return self.board[row][col], True
        return None, False

    def push_move(self, column):
        if not self.game_over and self.board[0][column] is None:
            new_board_obj = self.get_next_state(column)
            self.board = new_board_obj.board
            self.current_turn = new_board_obj.current_turn
            self.game_over = new_board_obj.game_over
            self.winner = new_board_obj.winner
            self.top_row_count = new_board_obj.top_row_count

    def get_result(self):
        return self.winner, self.game_over

    def get_square(self, row, col):
        return self.board[row][col]

    def get_inverse_board(self):
        new_board = self.copy()
        for row in range(6):
            for col in range(7):
                cur_cell = new_board.get_square(row, col)
                if not cur_cell is None:
                    new_board.board[row][col] = not cur_cell
        new_board.current_turn = not new_board.current_turn
        return new_board

    @property
    def legal_moves(self):
        for col in range(7):
            if self.board[0][col] is None:
                yield col

    def __str__(self):
        return_str = ""
        for row in range(6):
            for col in range(7):
                if self.board[row][col] is None:
                    return_str += "* "
                elif self.board[row][col]:
                    return_str += "R "
                else:
                    return_str += "Y "
                if col == 6:
                    return_str += "\n"
        return return_str

