import random

def play_game(game, agent1, agent2, i=-1, j=-1):
    agent1.exploration_rate = 0
    agent2.exploration_rate = 0
    state = game.reset()
    done = False

    turn = 0
    if i != -1 and j != -1:
        turn = 1
        game.step((i, j))

    while not done:
        current_agent = agent1 if turn % 2 == 0 else agent2
        valid_actions = game.get_valid_actions()
        action = current_agent.choose_action(state, valid_actions)
        state, _, done = game.step(action)

        turn += 1

    return game.get_winner()

class Evaluator:
    def __init__(self, game, agent1, agent2, random_first_move=False, episodes=1000):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.random_first_move = random_first_move
        self.episodes = episodes

    def evaluate(self):
        results = {
            "agent1": 0,
            "agent2": 0,
            "draw": 0
        }

        for i in range(self.episodes):
            if self.random_first_move:
                i = random.randint(0, 2)
                j = random.randint(0, 2)
            else:
                i = 0
                j = 0

            winner = play_game(self.game, self.agent1, self.agent2, i, j)

            if winner == 1:
                results["agent1"] += 1
            elif winner == 2:
                results["agent2"] += 1
            else:
                results["draw"] += 1

        print(f"Agent1 wins: {results['agent1']}")
        print(f"Agent2 wins: {results['agent2']}")
        print(f"Draws: {results['draw']}")

        return results