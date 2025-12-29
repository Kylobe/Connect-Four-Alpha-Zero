from AlphaConnectFour import AlphaZeroConnectFourNN, AlphaZero
from connect_four_env import ConnectFourEnv
import torch
from torch import optim
import random
from MCTS import MCTS
import numpy as np
import os
import multiprocessing as mp


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
        mcts.search(env.board)
        while not done:
            random_move = random.choice(list(env.board.legal_moves))
            state, done, winner = env.step(random_move)
            mcts.advance_root(random_move)
            print(state)
            print(f"Random Played: {random_move + 1}")
            if not done:
                action_probs = mcts.search(state)
                action = np.random.choice(7, p=action_probs)
                state, done, winner = env.step(action)
                mcts.advance_root(action)
                print(state)
                print(f"Alpha Zero Played: {action + 1}")
                if winner == False:
                    mcts_wins += 1
                    print("Alpha Zero Wins!")
            elif winner:
                random_wins += 1
                print("Random Wins!")
            else:
                print("Draw!")
        mcts.root = None
    print(f"Random Won: [{random_wins}/{n}], MCTS Won: [{mcts_wins}/{n}], Draws: [{n - (mcts_wins + random_wins)}/{n}]")

def mcts_vs_mcts(n):
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
    model2 = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model2.load_state_dict(torch.load("Phase1AlphaChess4.pt", map_location=torch.device('cpu')))

    model1 = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model1.load_state_dict(torch.load("Phase1AlphaChess0.pt", map_location=torch.device('cpu')))

    mcts1 = MCTS(args, model1)
    mcts2 = MCTS(args, model2)

    player1 = mcts1
    player2 = mcts2

    player1_is_mcts1 = True

    mcts1_wins = 0
    mcts2_wins = 0
    red_wins = 0
    yellow_wins = 0
    for _ in range(n):
        done = False
        env.reset()
        state = env.board
        player2.search(state)
        
        while not done:
            action_probs = player1.search(state)
            action = int(np.argmax(action_probs))
            state, done, winner = env.step(action)
            player1.advance_root(action)
            player2.advance_root(action)
            print(f"\nRed Played: {action + 1}")
            print(state)

            if not done:
                action_probs = player2.search(state)
                action = int(np.argmax(action_probs))
                state, done, winner = env.step(action)
                player1.advance_root(action)
                player2.advance_root(action)

                print(f"\nYellow Played: {action + 1}")
                print(state)
            if done:
                if winner:
                    red_wins += 1
                    if player1_is_mcts1:
                        mcts1_wins += 1
                        print("Older Model Wins!") 
                    else:
                        mcts2_wins += 1
                        print("Latest Model Wins!")
                elif winner == False:
                    yellow_wins += 1
                    if player1_is_mcts1:
                        mcts2_wins += 1
                        print("Latest Model Wins!")
                    else:
                        mcts1_wins += 1
                        print("Older Model Wins!") 
                elif done and winner is None:
                    print("Draw!")
        mcts1.root = None
        mcts2.root = None
        if player1_is_mcts1:
            player1 = mcts2
            player2 = mcts1
            player1_is_mcts1 = False
        else:
            player1 = mcts1
            player2 = mcts2
            player1_is_mcts1 = True
    print(f"Older Model Won: [{mcts1_wins}/{n}], Latest Model Won: [{mcts2_wins}/{n}], Red Won: [{red_wins}/{n}], Yellow Won: [{yellow_wins}/{n}] Draws: [{n - (mcts2_wins + mcts1_wins)}/{n}]")

def player_vs_mcts(n):
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

    model1 = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model1.load_state_dict(torch.load("Phase1AlphaChess4.pt", map_location=torch.device('cpu')))
    model1.share_memory()

    optimizer1 = optim.AdamW(model1.parameters(), lr=5e-3)
    optimizer1.load_state_dict(torch.load("LatestOptimizer.pt", map_location=torch.device('cpu')))

    mcts1 = MCTS(args, model1)

    mcts1_wins = 0
    player_wins = 0
    for _ in range(n):
        done = False
        env.reset()
        state = env.board
        mcts1.search(state)
        print(state)

        while not done:
            while True:
                try:
                    player_action = int(input("Select Column (1-7): ")) - 1
                    if player_action in state.legal_moves:
                        state, done, winner = env.step(player_action)
                        mcts1.advance_root(player_action)
                        break
                    else:
                        print(f"Illegal Move Chosen; Legal moves in this position: {[i + 1 for i in state.legal_moves]}")
                except Exception as e:
                    print(f"Exception: {e}")
                    print("Invalid Input. Please select an integer (1-7)")
            print(f"\nPlayer Played: {player_action + 1}")
            print(state)
            if not done:
                action_probs = mcts1.search(state)
                action = np.random.choice(7, p=action_probs)
                state, done, winner = env.step(action)
                mcts1.advance_root(action)
                print(f"\nLatest Model Played: {action + 1}")
                print(state)
                if winner == False:
                    mcts1_wins += 1
                    print("Alpha Zero Wins!")
            elif winner:
                player_wins += 1
                print("Player Wins!")
            else:
                print("Draw!")
        mcts1.root = None
    print(f"Player Won: [{player_wins}/{n}], Alpha Zero Won: [{mcts1_wins}/{n}], Draws: [{n - (player_wins + mcts1_wins)}/{n}]")

def get_majority_probability(prob_dist):
    tupled_probs = []
    for idx, prob in enumerate(prob_dist):
        tupled_probs.append((prob, idx))
    tupled_probs.sort(key=lambda x: x[0], reverse=True)
    THRESH_HOLD = 0.5
    new_probs = np.zeros(len(prob_dist), dtype=np.float32)
    total = 0
    for prob, idx in tupled_probs:
        if total < THRESH_HOLD:
            new_probs[idx] = prob
            total += prob
    new_probs /= new_probs.sum()
    return new_probs



# ---- worker ----
def _mcts_vs_mcts_worker(args_tuple):
    (
        games_to_play,
        args,
        older_path,
        newer_path,
        logging,
        worker_id,
    ) = args_tuple

    # Make each process deterministic-ish and not oversubscribe CPU
    torch.set_num_threads(1)
    np.random.seed(1234 + worker_id)

    # Create env + load models inside the worker
    env = ConnectFourEnv()

    # Force CPU for stability / simplicity in multiprocessing
    device = torch.device("cpu")

    model_newer = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model_newer.load_state_dict(torch.load(newer_path, map_location=device))
    model_newer.eval()

    model_older = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model_older.load_state_dict(torch.load(older_path, map_location=device))
    model_older.eval()

    mcts_older = MCTS(args, model_older)
    mcts_newer = MCTS(args, model_newer)

    # player1 goes first (Red). We’ll alternate which model is player1 each game.
    player1 = mcts_older
    player2 = mcts_newer
    player1_is_older = True

    older_wins = 0
    newer_wins = 0
    red_wins = 0
    yellow_wins = 0
    games_played = 0

    def log(msg=""):
        if logging:
            # prefix so you can see which process printed it
            print(f"{msg}", flush=True)

    for _ in range(games_to_play):
        games_played += 1
        done = False
        env.reset()
        state = env.board

        player2.search(state)

        while not done:
            # --- Red move (player1) ---
            action_probs = player1.search(state)
            action_probs = get_majority_probability(action_probs)
            action = np.random.choice(7, p=action_probs)
            state, done, winner = env.step(action)
            player1.advance_root(action)
            player2.advance_root(action)

            if logging:
                log(f"\nRed Played: {action + 1}")
                log("\n" + str(state))

            if done:
                break

            # --- Yellow move (player2) ---
            action_probs = player2.search(state)
            action_probs = get_majority_probability(action_probs)
            action = np.random.choice(7, p=action_probs)
            state, done, winner = env.step(action)
            player1.advance_root(action)
            player2.advance_root(action)

            if logging:
                log(f"\nYellow Played: {action + 1}")
                log("\n" + str(state))

        # Count results once per game
        if winner is True:
            red_wins += 1
            if player1_is_older:
                older_wins += 1
                log("Older Model Wins!")
            else:
                newer_wins += 1
                log("Newer Model Wins!")
        elif winner is False:
            yellow_wins += 1
            if player1_is_older:
                newer_wins += 1
                log("Newer Model Wins!")
            else:
                older_wins += 1
                log("Older Model Wins!")
        else:
            log("Draw!")

        # Reset tree between games
        mcts_older.root = None
        mcts_newer.root = None

        # Swap who is player1 next game (color balance)
        if player1_is_older:
            player1 = mcts_newer
            player2 = mcts_older
            player1_is_older = False
        else:
            player1 = mcts_older
            player2 = mcts_newer
            player1_is_older = True

    return older_wins, newer_wins, red_wins, yellow_wins, games_played


# ---- parallel driver ----
def mcts_vs_mcts_parallel(
    n: int,
    older_model_path: str,
    newer_model_path: str,
    logging: bool = False,
    num_processes: int = 4,
):
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

    # Split n across processes
    p = max(1, int(num_processes))
    base = n // p
    rem = n % p
    games_per_worker = [base + (1 if i < rem else 0) for i in range(p)]
    games_per_worker = [g for g in games_per_worker if g > 0]

    # On Windows, always use spawn
    ctx = mp.get_context("spawn")
    work = [
        (games_per_worker[i], args, older_model_path, newer_model_path, i == 0, i)
        for i in range(len(games_per_worker))
    ]

    with ctx.Pool(processes=len(games_per_worker)) as pool:
        results = pool.map(_mcts_vs_mcts_worker, work)

    older_wins = sum(r[0] for r in results)
    newer_wins = sum(r[1] for r in results)
    red_wins   = sum(r[2] for r in results)
    yellow_wins= sum(r[3] for r in results)
    played     = sum(r[4] for r in results)

    print(
        f"Older Model Won: [{older_wins}/{played}]\n"
        f"Newer Model Won: [{newer_wins}/{played}]\n"
        f"Red Won: [{red_wins}/{played}]\n"
        f"Yellow Won: [{yellow_wins}/{played}]\n"
        f"Draws: [{played - (older_wins + newer_wins)}/{played}]"
    )

    return older_wins, newer_wins, red_wins, yellow_wins, played




def main():
    mcts_vs_mcts_parallel(48, "LatestAlphaChess.pt", "Phase1AlphaChess4.pt", False, 12)




if __name__ == "__main__":
    main()

