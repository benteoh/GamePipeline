from games.tictactoe import TicTacToe
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.minimax_tictactoe_agent import MinimaxTicTacToeAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator

# Todos:
# reduce episodes,

def main():
    game = TicTacToe()
    # action_conversion = lambda action: action[0] * 3 + action[1]
    # action_back_conversion = lambda action: (action // 3, action % 3)
    # agent1 = PolicyGradientAgent(action_conversion=action_conversion, action_back_conversion=action_back_conversion)
    # agent2 = PolicyGradientAgent(action_conversion=action_conversion, action_back_conversion=action_back_conversion)

    agent1 = QLearningAgent(alpha=0.2, gamma=0.8, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05)
    agent2 = QLearningAgent(alpha=0.2, gamma=0.8, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05)

    
    trainer = Trainer(game, agent1, agent2, episodes=100000)
    trainer.train()

    # Check qvalues
    q_values = list(agent1.q_table.values())
    # print(q_values)
    print(f"Agent 1: Number of unlearned states: {q_values.count(0)} out of {len(q_values)}")

    q_values = list(agent2.q_table.values())
    # print(q_values)
    print(f"Agent 2: Number of unlearned states: {q_values.count(0)} out of {len(q_values)}")

    print("Training completed!")
    
    evaluator = Evaluator(game, agent1, agent2, random_first_move=True, episodes=1000)
    evaluator.evaluate()

    # Save the Q-tables
    # agent1.save("agent1.npy")
    # agent2.save("agent2.npy")
    
if __name__ == "__main__":
    main()
