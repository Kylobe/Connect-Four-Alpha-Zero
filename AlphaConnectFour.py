import torch
import torch.nn as nn
import torch.nn.functional as F
from MCTS import MCTS
from connect_four_env import ConnectFourEnv
import random
import numpy as np
from multiprocessing import Pool
from finite import finite_check
import time

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroConnectFourNN(nn.Module):
    def __init__(self, num_resBlocks, num_hidden):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.startBlock = nn.Sequential(
            nn.Conv2d(2, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True),
        )

        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])

        # --- Policy head: hidden -> 7 indices ---
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hidden, 32, kernel_size=1, padding=0, bias=True),
            nn.Flatten(1),
            nn.Linear(32 * 6 * 7, 7)
        )

        # --- Value head: matches the description more closely ---
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(1),              # keep batch dimension
            nn.Linear(6*7, 256),        # 42
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x):
        x = self.startBlock(x)
        for b in self.backBone:
            x = b(x)

        policy = self.policy_conv(x)
        value = self.value_head(x)
        return policy, value

class AlphaZero:
    def __init__(self, model, optimizer, env, args):
        self.model:AlphaZeroConnectFourNN = model
        self.optimizer = optimizer
        self.env:ConnectFourEnv = env
        self.args = args
        self.mcts = MCTS(args=args, model=model)

    def train(self, memory):
        random.shuffle(memory)
        total_losses = []
        policy_losses = []
        value_losses = []

        self.model.train()
        for batch_indx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batch_indx: batch_indx + self.args["batch_size"]]
            if not sample:
                continue

            state, policy_targets, value_targets, action_mask = zip(*sample)
            state = torch.stack(state).float().to(self.model.device)

            policy_targets = torch.tensor(
                np.array(policy_targets), dtype=torch.float32, device=self.model.device
            )

            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device
            )

            action_mask = torch.tensor(
                np.array(action_mask),
                dtype=torch.float32,
                device=self.model.device
            )

            out_policy, out_value = self.model(state)

            # Ensure targets only on legal moves, then renormalize
            policy_targets = policy_targets * action_mask
            policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # Mask illegal moves in logits with a huge negative number
            neg_inf = torch.finfo(out_policy.dtype).min
            masked_logits = out_policy.masked_fill(action_mask == 0, neg_inf)

            log_probs = F.log_softmax(masked_logits, dim=1)

            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return (
            sum(total_losses) / len(total_losses),
            sum(policy_losses) / len(policy_losses),
            sum(value_losses) / len(value_losses),
        )

    def self_play(self):
        assert_model_finite(self.model)
        memory = []
        player = True
        state = self.env.reset()
        self.mcts.root = None
        done = False
        turn = 0
        self.model.eval()
        while not done:
            turn += 1
            start = time.perf_counter()
            action_probs = self.mcts.search(state, num_searches=50*turn)
            memory.append((state, action_probs, player))
            action = np.random.choice(self.env.num_moves, p=action_probs)
            self.mcts.advance_root(action)
            state, done, result = self.env.step(action)
            end = time.perf_counter()
            print(f"Turn {turn} took: {end - start:.2f}s")
            if done:
                return_memory = []
                if result is None:
                    value = 0
                else:
                    value = 1
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    encoded = ConnectFourEnv.encode_board(hist_state)
                    action_mask = ConnectFourEnv.create_action_mask(hist_state)
                    return_memory.append(
                        (encoded, hist_action_probs, hist_outcome, action_mask)
                    )
                return return_memory
            player = not player

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()

            # Prepare shared model weights and args
            args_for_pool = [
                (self.model.state_dict(), self.args)
                for _ in range(self.args['num_processes'])
            ]

            with Pool(processes=self.args['num_processes']) as pool:
                results = pool.map(run_self_play, args_for_pool)

            for result in results:
                memory += result

            self.model.train()
            for epoch in range(self.args['epochs']):
                total_loss, policy_loss, value_loss = self.train(memory)
                print(f"Iteration: {iteration + 1}, Epoch: {epoch + 1}, Total Loss: {total_loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")

            torch.save(self.model.state_dict(), f"AlphaChess{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Optimizer{iteration}.pt")

    def self_play_wrapper(self):
        return AlphaZero().self_play()

def run_self_play(args):
    model_state_dict, alpha_args = args

    # Rebuild environment and model inside subprocess
    env = ConnectFourEnv()
    model = AlphaZeroConnectFourNN(alpha_args['res_blocks'], alpha_args['num_hidden'])
    model.load_state_dict(model_state_dict)
    model.eval()

    alpha = AlphaZero(model, None, env, alpha_args)
    return alpha.self_play()

def assert_model_finite(model):
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            raise RuntimeError(f"Param {name} has NaN/Inf")
    for name, b in model.named_buffers():
        # BatchNorm running_mean/running_var live in buffers
        if b is not None and not torch.isfinite(b).all():
            raise RuntimeError(f"Buffer {name} has NaN/Inf")

