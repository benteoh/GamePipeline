import random
from .base_agent import BaseAgent
from collections import defaultdict
import numpy as np

def to_hashable(state):
    """Converts a state into a hashable type (e.g., tuple)."""
    return tuple(state.flatten()) if isinstance(state, (list, np.ndarray)) else state

class MonteCarloAgent(BaseAgent):
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, gamma=0.99):
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.min_epsilon = min_epsilon  # Minimum epsilon
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # Q-table: key: (state, action), value: Q-value
        self.returns = defaultdict(list)  # Stores returns for each state-action pair

    def choose_action(self, state, valid_actions):
        """ Choose action based on epsilon-greedy strategy """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose the best action based on Q-values
            hashable_state = to_hashable(state)
            q_values = {action: self.q_table.get((hashable_state, action), 0) for action in valid_actions}
            max_q_value = max(q_values.values())
            best_actions = [action for action, q_value in q_values.items() if q_value == max_q_value]
            return random.choice(best_actions)

    def learn_from_episode(self, episode):
        """
        Learn using First-Visit Monte Carlo. Updates Q-values for all first visits in the episode.
        :param episode: List of tuples (state, action, reward)
        """
        visited = set()
        G = 0  # Initialize return

        # Iterate over the episode in reverse to calculate returns
        for state, action, reward in reversed(episode):
            hashable_state = to_hashable(state)
            state_action = (hashable_state, action)
            
            G = self.gamma * G + reward  # Update return
            
            # First-visit check
            if state_action not in visited:
                visited.add(state_action)
                self.returns[state_action].append(G)  # Store the return
                # Update Q-value as the mean of returns
                self.q_table[state_action] = np.mean(self.returns[state_action])

        # Decay epsilon after each episode to favor exploitation
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def learn(self, state, action, reward, next_state, done, valid_actions, eligibility_traces=None):
        """ Learn from the experience, updating the Q-table """
        pass