from connect_four import Board
from connect_four_env import ConnectFourEnv
import numpy as np
import torch
from finite import finite_check


class Node:
    def __init__(self, state: Board, terminal: bool, win_val, action_idx: int, prior, args, parent=None):
        self.state: Board = state
        self.terminal: bool = terminal
        self.win_val = win_val
        self.visit_count = 0
        self.win_count = 0
        self.action = action_idx
        self.prior = prior
        self.children = []
        self.parent = parent
        self.tensor_state = None
        self.args = args

    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        favorite_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                favorite_child = child
                best_ucb = ucb
        return favorite_child

    def get_ucb(self, child):
        # Terminals: child.win_val is from child's POV
        if child.terminal:
            if child.win_val == -1:   # child is lost => parent wins
                q_val = 1.0
            elif child.win_val == 1:  # child is winning => parent loses
                q_val = 0.0
            else:                     # draw
                q_val = 0.5
            u = 0.0  #no exploration on terminal nodes
        else:
            if child.visit_count > 0:
                ratio = child.win_count / child.visit_count  # in [-1,1]
                q_val = 1 - ((ratio + 1) / 2.0)              # parent success prob in [0,1]
            else:
                q_val = 0.5  # neutral

            u = self.args['C'] * (np.sqrt(self.visit_count + 1) / (child.visit_count + 1)) * child.prior

        return q_val + u

    def back_propagate(self, val):
        self.win_count += val
        self.visit_count += 1
        if self.parent is not None:
            self.parent.back_propagate(-val)

    def expand(self, policy, legal_actions):
        for action in legal_actions:
            prob = policy[action]
            temp_board: Board = self.state.copy()
            temp_board.push_move(action)
            result, child_terminal = temp_board.get_result()
            child_win_val = 0
            if child_terminal:
                if not result is None:
                    child_win_val = -1 if self.state.current_turn == result else 1
                    if child_win_val == -1 and not self.terminal:
                        self.terminal = True
                        self.win_val = 1
                        if not self.parent is None:
                            Node.back_prop_terminal(self.parent)
            child = Node(temp_board, child_terminal, child_win_val, action, prob, self.args, self)
            self.children.append(child)

    @staticmethod
    def back_prop_terminal(node):
        all_children_terminal = True
        all_children_winning = True
        for child in node.children:
            if not child.terminal:
                all_children_terminal = False
                all_children_winning = False
            elif child.win_val < 1:
                all_children_winning = False
        if all_children_winning:
            node.win_val = -1
            node.terminal = True
            if not node.parent is None:
                node.parent.win_val = 1
                node.terminal = True
                if not node.parent.parent is None:
                    Node.back_prop_terminal(node.parent.parent)
        elif all_children_terminal:
            node.win_val = 0
            node.terminal = True

class MCTS:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.root = None

    def create_root(self, state: Board):
        result, terminated = state.get_result()
        value = 0
        if terminated:
            value = 1 if result == state.current_turn else -1
        self.root = Node(state, terminated, value, 0, 0, self.args)

    def advance_root(self, action):
        if not self.root is None:
            for child in self.root.children:
                if action == child.action:
                    self.root = child
                    child.parent = None
                    break

    @torch.no_grad
    def search(self, state, num_searches = None):
        self.model.eval()
        if self.root is None:
            self.create_root(state)
        if num_searches is None:
            num_searches = self.args['num_searches']
        for _ in range(num_searches):
            if self.root is None:
                self.create_root(state)
            if self.root.terminal:
                mask = ConnectFourEnv.create_action_mask(self.root.state, self.model.device)
                finite_check("root terminal action mask", mask)
                if self.root.win_val == -1:
                    action_probs = np.ones(7, dtype=np.float32)
                    action_probs *= mask.detach().cpu().numpy()
                    action_probs /= np.sum(action_probs)
                    return action_probs
                elif self.root.win_val == 0:
                    action_probs = np.zeros(7, dtype=np.float32)
                    for child in self.root.children:
                        if child.win_val == 0:
                            action_probs[child.action] = 1.0
                            return action_probs
                else:
                    action_probs = np.zeros(7, dtype=np.float32)
                    for child in self.root.children:
                        if child.win_val == -1:
                            action_probs[child.action] = 1
                            return action_probs
            node = self.root
            while node.is_expanded() and not node.terminal:
                nxt = node.select()
                if nxt is None:
                    node.select()
                    break
                node = nxt

            value = node.win_val

            if not node.terminal:
                mask = ConnectFourEnv.create_action_mask(node.state, self.model.device)
                finite_check("action mask", mask)
                if node.tensor_state is None:
                    node.tensor_state = ConnectFourEnv.encode_board(node.state)
                state_input = node.tensor_state.unsqueeze(0)
                finite_check("tensor state", state_input)
                x = state_input.to(self.model.device)
                finite_check("x", x)
                policy_logits, value_t = self.model(x)
                finite_check("policy logits", policy_logits)

                finite_check("value logit", value_t)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0)
                finite_check("policy after softmax", policy)
                policy *= mask
                finite_check("policy after masking", policy)
                s = policy.sum()
                policy = policy / s if s > 0 else mask / mask.sum().clamp_min(1.0)
                finite_check("policy after normalizing", policy)
                policy_np = policy.detach().cpu().numpy()
                value = float(value_t.item())
                legal_actions = node.state.legal_moves
                node.expand(policy_np, legal_actions)

            node.back_propagate(value)

        action_probs = np.zeros(7)
        for child in self.root.children:
            child:Node
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


