from .base_agent import BaseAgent
from games.tictactoe import TicTacToe
import random
import math
import time

# Really slow for some reason.
class MinimaxTicTacToeAgent(BaseAgent):
    def __init__(self, player):
        self.player = player  # 1 or 2
        self.game = TicTacToe()

    def choose_action(self, state, valid_actions):
        if len(valid_actions) == self.game.size * self.game.size:
            # Random first move for simplicity (to avoid redundant calculation)
            return random.choice(valid_actions)

        self.game.reset()
        self.game.board = state
        _, move = self.minimax(self.game, self.player)
        return move

    def minimax(self, game, player):
        game.current_player = player
        max_player = self.player  # The agent is trying to maximize its score
        other_player = 3 - max_player

        # Base case: check for terminal state
        if game.get_winner() == max_player:
            return 1, None  # Agent wins
        elif game.get_winner() == other_player:
            return -1, None  # Opponent wins
        elif game.empty_squares() == 0:
            return 0, None  # Draw

        if player == max_player:
            best_score = -math.inf
            best_move = None
            for move in game.get_valid_actions():
                # Make the move
                game.do_action(player, move)
                score, _ = self.minimax(game, other_player)  # Recursive call
                game.undo_action(move)  # Undo the move
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

        else:  # Minimize for the opponent
            best_score = math.inf
            best_move = None
            for move in game.get_valid_actions():
                # Make the move
                game.do_action(player, move)
                score, _ = self.minimax(game, max_player)  # Recursive call
                game.undo_action(move)  # Undo the move
                game.current_winner = None  # Reset winner
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

    def learn(self, state, action, reward, next_state, done, valid_actions):
        pass

    def learn_from_episode(self, episode):
        pass