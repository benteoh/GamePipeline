from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state):
        """Choose an action given the current state."""
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done, eligibility_traces=None):
        """Learn from experience."""
        pass

    @abstractmethod
    def learn_from_episode(self, episode):
        """ Learn from experience. """
        pass
