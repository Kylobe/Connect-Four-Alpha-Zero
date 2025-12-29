import sys
import pygame
from connect_four import Board  # your Board class
from MCTS import MCTS
from AlphaConnectFour import AlphaZeroConnectFourNN
import torch
import numpy as np

# --------- CONFIG ----------
ROWS, COLS = 6, 7
CELL = 90               # pixel size of one cell
MARGIN_TOP = 80         # space for text/turn indicator
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + MARGIN_TOP
FPS = 60

# Colors
BG = (25, 25, 35)
GRID = (30, 90, 160)
EMPTY = (15, 15, 25)
RED = (220, 60, 60)
YELLOW = (240, 210, 70)
TEXT = (230, 230, 240)


def col_from_mouse_x(x: int) -> int:
    return max(0, min(COLS - 1, x // CELL))


def piece_color(cell):
    if cell is None:
        return EMPTY
    return RED if cell is True else YELLOW


def draw_board(screen, board: Board, font, msg=None):
    screen.fill(BG)

    # Header text
    if msg is None:
        turn_str = "Red" if board.current_turn else "Yellow"
        msg = f"Turn: {turn_str}   (Click a column)   R=restart  ESC=quit"
    text_surf = font.render(msg, True, TEXT)
    screen.blit(text_surf, (10, 20))

    # Draw grid background
    pygame.draw.rect(screen, GRID, pygame.Rect(0, MARGIN_TOP, WIDTH, ROWS * CELL))

    # Draw holes + pieces
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = MARGIN_TOP + r * CELL + CELL // 2

            cell = board.board[r][c]  # using your internal board storage
            color = piece_color(cell)

            pygame.draw.circle(screen, EMPTY, (cx, cy), CELL // 2 - 6)  # hole border
            pygame.draw.circle(screen, color, (cx, cy), CELL // 2 - 12)  # piece
    pygame.display.flip()


def animate_drop(screen, board_before: Board, board_after: Board, col: int, font):
    """
    Simple drop animation: find the row that changed in `col` and animate a falling piece.
    """
    # Identify new piece row
    changed_row = None
    for r in range(ROWS):
        if board_before.board[r][col] != board_after.board[r][col]:
            changed_row = r
            break
    if changed_row is None:
        return

    # The piece placed is in board_after at changed_row, col
    piece = board_after.board[changed_row][col]
    color = piece_color(piece)

    # Animate from top of grid down to target cell
    start_y = MARGIN_TOP + CELL // 2
    target_y = MARGIN_TOP + changed_row * CELL + CELL // 2
    x = col * CELL + CELL // 2

    clock = pygame.time.Clock()
    y = start_y
    vy = 0
    g = 2000  # gravity-ish

    while y < target_y:
        dt = clock.tick(FPS) / 1000.0
        vy += g * dt
        y += vy * dt

        # draw static board (before placement)
        draw_board(screen, board_before, font)

        # draw falling piece overlay
        pygame.draw.circle(screen, color, (x, int(min(y, target_y))), CELL // 2 - 12)
        pygame.display.flip()


def main():
    args = {
        'C': 2,
        'num_searches': 200,
        'num_iterations': 10,
        'num_self_play_iterations': 10,
        'epochs': 2,
        'num_processes': 10,
        'res_blocks': 40,
        'num_hidden': 256,
        'batch_size': 256,
        'base-lr': 0.0001,
    }
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 (pygame)")
    font = pygame.font.SysFont("consolas", 22)
    clock = pygame.time.Clock()

    board = Board()
    game_over = False
    end_msg = None

    device = torch.device("cpu")
    model = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model.load_state_dict(torch.load("Phase1AlphaChess4.pt", map_location=device))
    model.eval()

    mcts = MCTS(args, model)
    mcts.search(board)

    while True:
        clock.tick(FPS)

        # If game is over, show final message
        if game_over:
            draw_board(screen, board, font, end_msg)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    mcts.root = None
                    board = Board()
                    mcts.search(board)
                    game_over = False
                    end_msg = None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                x, _ = event.pos
                col = col_from_mouse_x(x)

                # validate move
                legal = set(board.legal_moves)  # assumes legal_moves yields valid cols
                if col not in legal:
                    # flash message briefly
                    draw_board(screen, board, font, "Illegal move. Pick a non-full column.")
                    pygame.time.delay(250)
                    continue

                # Apply move with animation
                before = board
                after = board.get_next_state(col)  # uses your method
                animate_drop(screen, before, after, col, font)

                board = after
                mcts.advance_root(col)

                print(mcts.root.state)

                winner, done = board.get_result()
                if not done:
                    draw_board(screen, board, font, "Alpha Zero Is Thinking")
                    action_probs = mcts.search(board)
                    action = int(np.argmax(action_probs))
                    before = board
                    after = board.get_next_state(action)
                    animate_drop(screen, before, after, action, font)
                    board = after
                    mcts.advance_root(action)
                    winner, done = board.get_result()
                if done:
                    mcts.root = None
                    game_over = True
                    if winner is None:
                        end_msg = "Draw!  Press R to restart  ESC to quit"
                    else:
                        end_msg = ("Red wins!" if winner is True else "Yellow wins!") + "  Press R to restart  ESC to quit"

        if not game_over:
            draw_board(screen, board, font)


if __name__ == "__main__":
    main()
