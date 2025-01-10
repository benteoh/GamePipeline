import random
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def choose_action(self, state, valid_actions):
        return random.choice(valid_actions)

    def learn(self, state, action, reward, next_state, done, valid_actions, eligibility_traces=None):
        pass