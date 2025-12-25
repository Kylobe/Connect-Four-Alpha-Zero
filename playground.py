from connect_four import Board
from MCTS import MCTS
from connect_four_env import ConnectFourEnv
from AlphaConnectFour import AlphaZeroConnectFourNN, AlphaZero
import torch
from torch import optim
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingLR


def mcts_playground():
    load_model = False
    args = {
        'C': 2,
        'num_searches': 1000,
        'num_iterations': 10,
        'num_self_play_iterations': 5,
        'epochs': 1,
        'num_processes': 1,
        'res_blocks': 40,
        'num_hidden': 256,
        'batch_size': 256,
        'base-lr': 0.1,
    }

    env = ConnectFourEnv()
    model = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    if load_model:
        model.load_state_dict(torch.load("LatestAlphaChess.pt", map_location=torch.device('cpu')))
    model.share_memory()



    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    if load_model:
        optimizer.load_state_dict(torch.load("LatestOptimizer.pt", map_location=torch.device('cpu')))
    mcts = MCTS(args, model)
    b = Board()
    inputs = [1, 6, 2, 6]
    for action in inputs:
        b.push_move(action)
    probs = mcts.search(b)
    print(probs)


def main():
    b = Board()
    inputs = [1, 6, 2, 6]
    for action in inputs:
        b.push_move(action)
    print(b)
    print(b.get_result())

if __name__ == "__main__":
    main()
    mcts_playground()