from abc import ABC, abstractmethod

class BaseZeroSumGame(ABC):
    @abstractmethod
    def reset(self):
        """Reset the game to the initial state."""
        pass

    @abstractmethod
    def get_state(self):
        """Return the current game state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply an action for the current player."""
        pass

    @abstractmethod
    def do_action(self, player, action):
        """Apply an action for a player."""
        pass

    @abstractmethod
    def get_valid_actions(self):
        """Return a list of valid moves for the current state."""
        pass

    @abstractmethod
    def is_game_over(self):
        """Check if the game is over."""
        pass

    @abstractmethod
    def get_winner(self):
        """Return the winner, or None if the game is a draw."""
        pass
