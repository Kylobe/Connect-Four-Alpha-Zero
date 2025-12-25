from AlphaConnectFour import AlphaZeroConnectFourNN, AlphaZero
from connect_four_env import ConnectFourEnv
import torch
from torch import optim
import random
from MCTS import MCTS
import numpy as np


def random_vs_mcts(n: int):
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

    env = ConnectFourEnv()
    model = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model.load_state_dict(torch.load("LatestAlphaChess.pt", map_location=torch.device('cpu')))
    model.share_memory()

    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    optimizer.load_state_dict(torch.load("LatestOptimizer.pt", map_location=torch.device('cpu')))
    alpha = AlphaZero(model, optimizer, env, args)
    random_wins = 0
    mcts_wins = 0
    mcts = MCTS(args, model)
    for _ in range(n):
        done = False
        env.reset()
        while not done:
            random_move = random.choice(list(env.board.legal_moves))
            state, done, winner = env.step(random_move)
            if not done:
                action_probs = mcts.search(state)
                action = np.random.choice(7, p=action_probs)
                state, done, winner = env.step(action)
                if winner == False:
                    mcts_wins += 1
            elif winner:
                random_wins += 1
    print(f"Random Won: [{random_wins}/{n}], MCTS Won: [{mcts_wins}/{n}], Draws: [{n - (mcts_wins + random_wins)}/{n}]")


def main():
    random_vs_mcts()



if __name__ == "__main__":
    main()

