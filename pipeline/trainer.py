import numpy as np

def to_hashable(state):
    """Converts a state into a hashable type (e.g., tuple)."""
    return tuple(state.flatten()) if isinstance(state, (list, np.ndarray)) else state

class Trainer:
    def __init__(self, game, agent1, agent2, episodes=1000):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = episodes

    def train(self):
        for episode in range(self.episodes):
            state = self.game.reset()
            done = False
            turn = 0

            agent1_memory = []
            agent2_memory = []

            while not done:
                current_agent = self.agent1 if turn % 2 == 0 else self.agent2
                valid_actions = self.game.get_valid_actions()
                action = current_agent.choose_action(state, valid_actions)

                # Remove the chosen action from the valid actions
                valid_actions.remove(action)

                # Make a move.
                next_state, _, done = self.game.step(action)

                # For temporal difference agents
                self.agent1.learn(state, action, self.game.reward(1), next_state, done, valid_actions)
                self.agent2.learn(state, action, self.game.reward(2), next_state, done, valid_actions)

                agent1_memory.append((state, action, self.game.reward(1), next_state, done))
                agent2_memory.append((state, action, self.game.reward(2), next_state, done))

                # Update state and alternate turns
                state = next_state
                turn += 1

            # For agents that learn from whole episodes
            self.agent1.learn_from_episode(agent1_memory)
            self.agent2.learn_from_episode(agent2_memory)

            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.episodes} - Agent 1 reward: {self.game.reward(1)} - Agent 2 reward: {self.game.reward(2)}")
