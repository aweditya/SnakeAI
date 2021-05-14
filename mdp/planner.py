#! /usr/bin/python
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()


class Plan:
    def __init__(self, mdp, algorithm):
        self.states = 5
        self.actions = 2
        self.gamma = 0.9
        self.start = 0
        self.end = []
        self.state_transition_probabilities = np.zeros(
            (self.states, self.actions, self.states))
        self.expected_reward = np.zeros((self.states, self.actions))

        self.process_input(mdp)
        if algorithm == "vi":
            self.value_iteration()

    def process_input(self, mdp):
        data = open(mdp, 'r')
        for line in data:
            elements = line.split(' ')
            if elements[0] == "numStates":
                self.states = int(elements[1])
            elif elements[0] == "numActions":
                self.actions = int(elements[1])
                self.state_transition_probabilities = np.zeros(
                    (self.states, self.actions, self.states))
                self.expected_reward = np.zeros((self.states, self.actions))
            elif elements[0] == "end":
                self.end.append(list(map(int, elements[1:])))
            elif elements[0] == "transition":
                initial_state = int(elements[1])
                action = int(elements[2])
                final_state = int(elements[3])
                reward = float(elements[4])
                probability = float(elements[5])
                self.state_transition_probabilities[initial_state][action][
                    final_state] += probability
                self.expected_reward[initial_state][
                    action] += probability * reward
            elif elements[0] == "discount":
                self.gamma = float(elements[2])

    def value_iteration(self):
        self.state_value_function = np.zeros((self.states, 1, 1))
        self.optimal_policy = np.zeros((self.states, 1, 1))

        theta = 1e-28
        delta = 1
        while delta > theta:
            old_state_value_function = self.state_value_function
            self.state_value_function = np.max(
                self.expected_reward +
                self.gamma * np.sum(self.state_transition_probabilities *
                                    self.state_value_function,
                                    axis=2),
                axis=1)
            error = np.sum(
                np.linalg.norm(old_state_value_function -
                               self.state_value_function))
            delta = min(error, delta)

        self.optimal_policy = np.argmax(
            self.expected_reward +
            self.gamma * np.sum(self.state_transition_probabilities *
                                self.state_value_function,
                                axis=2),
            axis=1)

    def print_result(self):
        for state in range(self.states):
            print(self.state_value_function[state], self.optimal_policy[state])


if __name__ == "__main__":
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str, default="vi")

    args = parser.parse_args()
    if not (args.algorithm == "vi" or args.algorithm == "hpi"
            or args.algorithm == "lp"):
        print("Algorithm not supported")
        sys.exit(0)

    algo = Plan(args.mdp, args.algorithm)
    algo.print_result()
