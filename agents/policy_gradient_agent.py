import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent

class PolicyGradientAgent(BaseAgent):
    class PolicyModel(nn.Module):
        def __init__(self, input_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.softmax(self.fc2(x), dim=-1)

    def __init__(self, input_dim=9, action_dim=9, learning_rate=0.1, gamma=0.8, action_conversion=None, action_back_conversion=None):
        """
        Initializes the policy gradient agent.
        Args:
            input_dim: Number of state features.
            action_dim: Number of possible actions.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor for rewards.
        """
        self.model = self.PolicyModel(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.action_conversion = action_conversion
        self.action_back_conversion = action_back_conversion

    def choose_action(self, state: np.ndarray, valid_actions: list) -> int:
        """
        Choose an action based on the current policy (model).
        Args:
            state: Tensor representing the current state.
        Returns:
            int: Chosen action.
        """
        state = torch.tensor(state.flatten(), dtype=torch.float32)

        with torch.no_grad():
            action_prob = self.model(state)

            # Eliminate invalid actions
            valid_actions = [self.action_conversion(action) for action in valid_actions]
            for i in range(len(action_prob)):
                if i not in valid_actions:
                    action_prob[i] = 0.0

            # If all actions are invalid, choose randomly
            if action_prob.sum() == 0:
                action_prob = torch.tensor([1.0 if i in valid_actions else 0.0 for i in range(len(action_prob))])

            action_prob /= action_prob.sum()
            action = torch.multinomial(action_prob, num_samples=1).item()
        return self.action_back_conversion(action)

    def learn(self, state, action, reward, next_state, done, eligibility_traces=None):
        """
        This agent doesn't learn step-by-step, so this method can raise a NotImplementedError.
        """
        pass

    def learn_from_episode(self, episode):
        """
        Learn from an episode's data.
        Args:
            episode: A list of tuples (state, action, reward, next_state, done).
        """
        states = []
        actions = []
        rewards = []

        for state, action, reward, _, _ in episode:
            states.append(torch.tensor(state.flatten(), dtype=torch.float32))
            actions.append(self.action_conversion(action))
            rewards.append(reward)

        # Compute discounted rewards
        discounted_rewards = []
        reward_sum = 0.0
        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards = discounted_rewards[::-1]

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Prepare data
        states = torch.vstack(states)
        actions = torch.tensor(actions)

        # Compute loss
        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, actions) * discounted_rewards
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()