#! /usr/bin/python
import argparse
import sys
import numpy as np
from pulp import *

parser = argparse.ArgumentParser()


class Plan:
    def __init__(self, mdp, algorithm):
        self.states = 5
        self.actions = 2
        self.gamma = 0.9
        self.start = 0
        self.end = []
        self.state_transition_probabilities = np.zeros((self.states, self.actions, self.states)) # p(s'|s,a)
        self.expected_reward = np.zeros((self.states, self.actions)) # r(s,a)

        self.process_input(mdp)
        if algorithm == "hpi":
            state_value_function, optimal_policy = self.howard_policy_iteration()
            self.print_result(state_value_function, optimal_policy)

        elif algorithm == "vi":
            state_value_function, optimal_policy = self.value_iteration()
            self.print_result(state_value_function, optimal_policy)

        elif algorithm == "lp":
            state_value_function, optimal_policy = self.linear_programming()
            self.print_result(state_value_function, optimal_policy)

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
                self.state_transition_probabilities[initial_state][action][final_state] += probability
                self.expected_reward[initial_state][action] += probability * reward
            elif elements[0] == "discount":
                self.gamma = float(elements[2])

    def howard_policy_iteration(self):
        state_value_function = np.zeros(self.states)
        optimal_policy = np.zeros(self.states, dtype=int)

        while True:
            # policy evaluation
            A = np.eye(self.states) - self.gamma * self.state_transition_probabilities[np.arange(self.states), optimal_policy, :]
            b = self.expected_reward[np.arange(self.states), optimal_policy]
            state_value_function = np.linalg.solve(A, b)

            # policy improvement
            current_best_policy = np.argmax(self.expected_reward + self.gamma * self.state_transition_probabilities @ state_value_function, axis=1)
            if np.array_equal(current_best_policy, optimal_policy):
                optimal_policy = current_best_policy
                break
            else:
                optimal_policy = current_best_policy

        return state_value_function, optimal_policy            

    def value_iteration(self):
        state_value_function = np.zeros((self.states, 1, 1))
        optimal_policy = np.zeros((self.states, 1, 1))

        while True:
            old_state_value_function = state_value_function
            state_value_function = np.max(self.expected_reward + self.gamma * np.sum(self.state_transition_probabilities * state_value_function, axis=2), axis=1)
            error = np.sum(np.linalg.norm(old_state_value_function - state_value_function))

            if np.allclose(old_state_value_function, state_value_function, rtol=1e-11, atol=1e-08):
                break

        optimal_policy = np.argmax(self.expected_reward + self.gamma * np.sum(self.state_transition_probabilities * state_value_function, axis=2), axis=1)

        return state_value_function, optimal_policy

    def linear_programming(self):
        state_value_function = np.zeros(self.states)

        # Define the LP problem in PuLP to maximize -sum(V(s)) or minimize sum(V(s))
        prob = LpProblem("Bellman Optimality", LpMinimize)

        lpVariables = []
        for i in range(self.states):
            variable = LpVariable(f"V_{i}")
            lpVariables.append(variable)

        # Add the function to be maximised or minimised
        prob += sum(lpVariables)

        # Add the nk constraints where n is the number of states and k is the number of actions
        for i in range(self.states):
            for j in range(self.actions):
                constraint = self.expected_reward[i][j]
                for k in range(self.states):
                    constraint += self.gamma * self.state_transition_probabilities[i][j][k] * lpVariables[k]

                prob += lpVariables[i] >= constraint

        prob.solve(PULP_CBC_CMD(msg=0))

        for i in range(self.states):
            state_value_function[i] = lpVariables[i].varValue

        optimal_policy = np.argmax(self.expected_reward + self.gamma * np.sum(self.state_transition_probabilities * state_value_function, axis=2), axis=1)

        return state_value_function, optimal_policy        

    def print_result(self, state_value_function, optimal_policy):
        for state in range(self.states):
            print(state_value_function[state], optimal_policy[state])


if __name__ == "__main__":
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str, default="vi")

    args = parser.parse_args()
    if not (args.algorithm == "vi" or args.algorithm == "hpi"
            or args.algorithm == "lp"):
        print("Algorithm not supported")
        sys.exit(0)

    algo = Plan(args.mdp, args.algorithm)
