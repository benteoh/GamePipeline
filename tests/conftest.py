import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".", ".")))

from games.tictactoe import TicTacToe

@pytest.fixture
def tictactoe_game(size=3):
    """Fixture to initialize a TicTacToe game."""
    return TicTacToe(size)
