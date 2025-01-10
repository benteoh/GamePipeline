def test_reset(tictactoe_game):
    """Test if reset initializes the board correctly."""
    state = tictactoe_game.reset()
    assert state.shape == (3, 3)
    assert (state == 0).all()

def test_get_valid_moves(tictactoe_game):
    """Test if valid moves are correctly identified."""
    tictactoe_game.reset()
    moves = tictactoe_game.get_valid_moves()
    assert len(moves) == 9
    assert all(isinstance(move, tuple) for move in moves)

    # Make a move and check valid moves again
    tictactoe_game.make_move(1, (0, 0))
    moves = tictactoe_game.get_valid_moves()
    assert len(moves) == 8
    assert (0, 0) not in moves

def test_make_move(tictactoe_game):
    """Test making moves and board state updates."""
    tictactoe_game.reset()

    # Player 1 makes a move
    state, reward, done = tictactoe_game.make_move(1, (0, 0))
    assert state[0, 0] == 1
    assert reward == 0
    assert not done

    # Player 2 makes a move
    state, reward, done = tictactoe_game.make_move(2, (1, 1))
    assert state[1, 1] == 2
    assert reward == 0
    assert not done

    # Player 1 tries an invalid move
    with pytest.raises(ValueError, match="Invalid move: Cell already occupied."):
        tictactoe_game.make_move(1, (0, 0))

def test_game_over_conditions(tictactoe_game):
    """Test game over conditions."""
    tictactoe_game.reset()

    # Simulate a winning condition for Player 1
    tictactoe_game.make_move(1, (0, 0))
    tictactoe_game.make_move(2, (1, 0))
    tictactoe_game.make_move(1, (0, 1))
    tictactoe_game.make_move(2, (1, 1))
    state, reward, done = tictactoe_game.make_move(1, (0, 2))

    assert done
    assert reward == 1
    assert tictactoe_game.get_winner() == 1

    # Reset and simulate a draw
    tictactoe_game.reset()
    moves = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 0), (1, 2),
        (2, 0), (2, 2), (2, 1)
    ]
    for i, move in enumerate(moves):
        tictactoe_game.make_move(1 if i % 2 == 0 else 2, move)

    assert tictactoe_game.is_game_over()
    assert tictactoe_game.get_winner() is None
