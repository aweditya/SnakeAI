#! /usr/bin/python
import argparse
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("--algorithm", type=str, default="sarsa")
    parser.add_argument("--moves", type=str, default="normal")
    parser.add_argument("--stochastic", type=int, default=0)
    args = parser.parse_args()

    if args.algorithm == "compare":
        sarsa_agent = Agent(moves=args.moves, stochastic=args.stochastic)
        sarsa_progress = sarsa_agent.train(algorithm="sarsa")

        q_learning_agent = Agent(moves=args.moves, stochastic=args.stochastic)
        q_learning_progress = q_learning_agent.train(algorithm="q-learning")

        expected_sarsa_agent = Agent(moves=args.moves, stochastic=args.stochastic)
        expected_sarsa_progress = expected_sarsa_agent.train(algorithm="expected-sarsa")

        dyna_q_agent = Agent(moves=args.moves, stochastic=args.stochastic)
        dyna_q_progress = dyna_q_agent.train(algorithm="dyna-q")

        plt.plot(sarsa_progress, np.arange(1, len(sarsa_progress) + 1), label="SARSA(0)")
        plt.plot(q_learning_progress, np.arange(1, len(q_learning_progress) + 1), label="Q-Learning")
        plt.plot(expected_sarsa_progress, np.arange(1, len(expected_sarsa_progress) + 1), label="Expected-SARSA")
        plt.plot(dyna_q_progress, np.arange(1, len(dyna_q_progress) + 1), label="Dyna-Q")
        plt.title(f"Comparison of algorithms. {args.moves.title()} moves; Stochastic wind = {bool(args.stochastic)}")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.legend()
        plt.grid()
        plt.show()    

    else:
        agent = Agent(moves=args.moves, stochastic=args.stochastic)
        progress = agent.train(algorithm=args.algorithm)    
        plt.plot(progress, np.arange(1, len(progress) + 1))
        plt.title(f"{args.algorithm.title()} with {args.moves.title()} moves and Stochastic wind = {bool(args.stochastic)}")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.grid()
        plt.show()