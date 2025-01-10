import random
import numpy as np
from .base_agent import BaseAgent

def to_hashable(state):
    """Converts a state into a hashable type (e.g., tuple)."""
    return tuple(state.flatten()) if isinstance(state, (list, np.ndarray)) else state

class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.2, gamma=0.8, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.min_epsilon = min_epsilon  # Minimum epsilon
        self.q_table = {}  # Q-table: key: (state, action), value: Q-value

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

    def learn(self, state, action, reward, next_state, done, valid_actions, eligibility_traces=None):
        """ Learn from the experience, updating the Q-table """
        hashable_state = to_hashable(state)
        hashable_next_state = to_hashable(next_state)

        next_max = 0 if done else max(self.q_table.get((hashable_next_state, a), 0) for a in valid_actions)
        target = reward + self.gamma * next_max
        
        # Update Q-value using the temporal difference formula
        current_q_value = self.q_table.get((hashable_state, action), 0)
        eligibility_mult = 1 if eligibility_traces is None else eligibility_traces[(hashable_state, action)]
        self.q_table[(hashable_state, action)] = current_q_value + self.alpha * (target - current_q_value) * eligibility_mult

        # Decay epsilon after each episode
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def learn_from_episode(self, episode):
        """ Learn from experience. """
        pass

    def save(self, filename):
        """ Save the Q-table to a file. """
        np.save(filename, self.q_table)

    def load(self, filename):
        """ Load the Q-table from a file. """
        self.q_table = np.load(filename, allow_pickle=True).item()