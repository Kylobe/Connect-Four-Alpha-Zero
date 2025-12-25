import pytest
from connect_four import Board

# Assume Board is imported from your module:
# from your_module import Board


def set_cells(b: Board, cells, value):
    """
    cells: iterable of (row, col)
    value: True (R), False (Y)
    """
    for r, c in cells:
        b.board[r][c] = value


def fill_board_draw_no_wins(b: Board):
    """
    Fills the board completely without creating:
      - any vertical 4-in-a-row
      - any horizontal 4-in-a-row
      - any down-right diagonal 4-in-a-row (the only diagonal your get_result checks)

    Pattern idea:
      - Use 3-in-a-row blocks horizontally: R R R Y Y Y R (or flipped)
      - Alternate the pattern each row to break diagonals
    """
    row_patterns = [
        # R=True, Y=False
        [True,  True,  True,  False, False, False, True],
        [False, False, False, True,  True,  True,  False],
        [True,  True,  True,  False, False, False, True],
        [False, False, False, True,  True,  True,  False],
        [True,  True,  True,  False, False, False, True],
        [False, False, False, True,  True,  True,  False],
    ]

    for r in range(6):
        for c in range(7):
            b.board[r][c] = row_patterns[r][c]


# --------------------------
# Basic / no result
# --------------------------

def test_get_result_initial_board_not_terminal():
    b = Board()
    winner, terminal = b.get_result()
    assert winner is None
    assert terminal is False

@pytest.mark.parametrize(
        "input,expected",
        [
            ([], (None, False)),
            ([0, 1, 0, 1, 0, 1, 0], (True, True)),
            ([0, 1, 0, 1, 0, 1, 2, 1], (False, True)),
            ([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3], (True, True)),
            ([0, 0, 1, 1, 2, 2, 3], (True, True)),
            ([0, 1, 1, 2, 2, 3, 3, 3, 3, 6, 2], (True, True)),
            ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 6, 3, 3, 3, 3, 3, 6, 3], (False, True)),
            ([6 for _ in range(6)] + [5 for _ in range(6)] + [4 for _ in range(6)] + [0] + [3 for _ in range(5)] + [0, 3], (False, True)),
            ([0 for _ in range(4)] + [6] + [1 for _ in range(3)] + [2 for _ in range(2)] + [6, 3], (False, True)),
            ([0 for _ in range(6)] + [1 for _ in range(6)] + [2 for _ in range(6)] + [6] + [3 for _ in range(6)] + [4 for _ in range(6)] + [5 for _ in range(6)] + [6 for _ in range(5)], (None, True)),
            ([3,2,2,1,1,0,1,0,0,6,0], (True, True)),
            ([3,2,1,0,0,0,0,1,1,6,2], (True, True)),
            ([6,3,2,1,0,0,0,0,1,1,6,2], (False, True)),
            ([0,1,0,1,2,1,3,1], (False, True)),
            ([0,0,1,1,3,3,2], (True, True)),
            ([0,0,1,1,2,2,3,4], (True, True))
        ]
)
def test_connect_four_move_sequences(input, expected):
    b = Board()
    for action in input:
        b.push_move(action)
    expected_winner, expected_terminal = expected
    winner, terminal = b.get_result()
    assert winner == expected_winner
    assert terminal == expected_terminal
