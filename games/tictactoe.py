from games.base_zero_sum_game import BaseZeroSumGame
import numpy as np

class TicTacToe:
    def __init__(self, size=3):
        assert size > 2, "Board size must be greater than 2"
        self.size = size
        self.board = np.zeros((size, size), dtype=int) # 0: Empty, 1: X, 2: O
        self.current_player = 1

    def display_board(self):
        """Display the current board state."""
        symbol = lambda x: 'X' if x == 1 else 'O' if x == 2 else ' '
        for i in range(self.size):
            print(' | '.join(([symbol(x) for x in self.board[i]])))
        print()

    def reset(self):
        """Reset the board to its initial state."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """Return the current board state."""
        return self.board.copy()

    def step(self, action):
        """Apply an action for the current player."""
        state, reward, done = self.do_action(self.current_player, action)
        self.current_player = 3 - self.current_player  # Switch players
        return state, reward, done

    def do_action(self, player, action):
        """Apply an action for the given player."""
        if player != self.current_player:
            raise ValueError("Invalid player: Not their turn.")

        x, y = action
        if self.board[x, y] != 0:
            raise ValueError("Invalid action: Cell already occupied.")

        self.board[x, y] = player

        return self.get_state(), self.reward(player), self.is_game_over()

    def get_valid_actions(self):
        """Return a list of valid moves as (x, y) tuples."""
        return [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x, y] == 0]

    def is_game_over(self):
        """Check if the game is over."""
        if self.get_winner() is not None:
            return True
        return np.all(self.board != 0)  # Game is over if all cells are filled

    def get_winner(self):
        """Return the winner of the game."""
        for player in [1, 2]:
            # Check rows and columns
            if np.any(np.all(self.board == player, axis=0)) or np.any(np.all(self.board == player, axis=1)):
                return player

            # Check diagonals
            if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        return None

    def empty_squares(self):
        """Return the number of empty squares."""
        return np.sum(self.board == 0)

    def undo_action(self, action):
        """Undo the last move."""
        x, y = action
        if self.board[x, y] == 0:
            raise ValueError("Invalid action: Cell is already empty.")
        
        self.current_player = 3 - self.current_player
        
        self.board[x, y] = 0

    def reward(self, player):
        """Return the reward for the given player."""
        winner = self.get_winner()
        if winner is None:
            return 0
        return 1 if winner == player else -1

