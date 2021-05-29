#! /usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy as np
import random


class NormalWindyGridworld:
    def __init__(self):
        self.rows = 7
        self.cols = 10
        self.start = 3 * self.cols + 0  # (3, 0)
        self.end = 3 * self.cols + 7  # (3, 7)

        # how far is the agent pushed upwards on making a move
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def convert_2D_to_1D(self, x, y):
        return x * self.cols + y

    def convert_1D_to_2D(self, state):
        return state // self.cols, state % self.cols

    def epsilon_greedy(self, state, action_value_function, epsilon,
                       numActions):
        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action = np.random.randint(0, numActions)
        else:
            action = np.random.choice(
                np.flatnonzero(action_value_function[state, :] ==
                               action_value_function[state, :].max()))

        return action

    def normal_move(self, x, y, action):
        # 0-N, 1-E, 2-S, 3-W
        if (action == 0):
            next_x = max(x - 1, 0)
            next_y = y

        elif (action == 1):
            next_x = x
            next_y = min(y + 1, self.cols - 1)

        elif (action == 2):
            next_x = min(x + 1, self.rows - 1)
            next_y = y

        elif (action == 3):
            next_x = x
            next_y = max(y - 1, 0)

        next_x = max(next_x - self.wind_strength[y], 0)
        return next_x, next_y

    def sarsa(self, episodes=170, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 4

        data = []

        action_value_function = np.zeros((numStates, numActions))
        for episode in range(episodes):
            currentState = self.start
            action = self.epsilon_greedy(currentState, action_value_function,
                                         epsilon, numActions)
            while (currentState != self.end):
                data.append(episode)
                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.normal_move(x, y, action)

                nextState = self.convert_2D_to_1D(next_x, next_y)
                nextAction = self.epsilon_greedy(nextState,
                                                 action_value_function,
                                                 epsilon, numActions)

                error = reward + gamma * \
                    action_value_function[nextState, nextAction] - \
                    action_value_function[currentState, action]
                action_value_function[currentState, action] += alpha * error
                currentState = nextState
                action = nextAction

        plt.title('Normal Windy Gridworld using SARSA(0)')
        self.plot_learning_curve(data)

        return action_value_function

    def q_learning(self, episodes=170, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 4

        data = []

        action_value_function = np.zeros((numStates, numActions))
        for episode in range(episodes):
            currentState = self.start
            while (currentState != self.end):
                # print(episode)
                data.append(episode)

                action = self.epsilon_greedy(currentState,
                                             action_value_function, epsilon,
                                             numActions)

                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.normal_move(x, y, action)

                nextState = self.convert_2D_to_1D(next_x, next_y)

                error = reward + gamma * \
                    np.max(action_value_function[nextState, :]) - \
                    action_value_function[currentState, action]
                action_value_function[currentState, action] += alpha * error
                currentState = nextState

        plt.title('Normal Windy Gridworld using Q-learning')
        self.plot_learning_curve(data)

        return action_value_function

    def current_policy(self, state, action_value_function, epsilon, numActions):
        policy = np.ones((1, numActions)) * epsilon / numActions
        greedy_action = np.random.choice(
            np.flatnonzero(action_value_function[state, :] ==
                           action_value_function[state, :].max()))
        policy[0, greedy_action] += 1 - epsilon

        return policy

    def expected_sarsa(self, episodes=170, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 4

        data = []

        action_value_function = np.zeros((numStates, numActions))
        for episode in range(episodes):
            currentState = self.start
            while (currentState != self.end):
                # print(episode)
                data.append(episode)

                action = self.epsilon_greedy(currentState,
                                             action_value_function, epsilon,
                                             numActions)

                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.normal_move(x, y, action)

                nextState = self.convert_2D_to_1D(next_x, next_y)

                error = reward + gamma * \
                    np.sum(self.current_policy(nextState, action_value_function, epsilon, numActions) *
                           action_value_function[nextState, :]) - action_value_function[currentState, action]
                action_value_function[currentState, action] += alpha * error
                currentState = nextState

        plt.title('Normal Windy Gridworld using Expected Sarsa')
        self.plot_learning_curve(data)

        return action_value_function

    def plot_learning_curve(self, data):
        plt.xlabel('Time Steps')
        plt.ylabel('Episodes')
        plt.plot(data)

        plt.show()

    def dyna_q(self, episodes=50, gamma=0.95, epsilon=0.1, alpha=0.1, n=50):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 4

        steps_per_episode = []

        action_value_function = np.zeros((numStates, numActions))
        model = np.zeros((numStates, numActions, 2))
        previously_observed_state = np.zeros((numStates, 1))
        previously_observed_action = np.zeros((numStates, numActions))

        for episode in range(episodes):
            currentState = self.start
            steps = 0
            while (currentState != self.end):
                # print(steps)
                steps += 1
                previously_observed_state[currentState, 0] = 1
                action = self.epsilon_greedy(currentState,
                                             action_value_function, epsilon,
                                             numActions)
                previously_observed_action[currentState, action] = 1

                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.normal_move(x, y, action)

                nextState = self.convert_2D_to_1D(next_x, next_y)

                error = reward + gamma * np.max(action_value_function[
                    nextState, :]) - action_value_function[currentState,
                                                           action]
                action_value_function[currentState, action] += alpha * error
                model[currentState, action, 0] = reward
                model[currentState, action, 1] = nextState

                currentState = nextState

                for i in range(n):
                    random_state = int(
                        np.random.choice(
                            np.flatnonzero(previously_observed_state[:,
                                                                     0] == 1)))
                    random_action = int(
                        np.random.choice(
                            np.flatnonzero(previously_observed_action[
                                random_state, :] == 1)))
                    reward = model[random_state, random_action, 0]
                    next_state = int(model[random_state, random_action, 1])
                    error = reward + gamma * np.max(action_value_function[
                        next_state, :]) - action_value_function[random_state,
                                                                random_action]
                    action_value_function[random_state,
                                          random_action] += alpha * error

            steps_per_episode.append(steps)

        plt.title('Normal Windy Gridworld using Dyna-Q')
        self.plot_steps_per_episode(steps_per_episode)

        return action_value_function

    def plot_steps_per_episode(self, steps_per_episode):
        plt.xlabel('Episodes')
        plt.ylabel('Steps per episode')
        plt.plot(steps_per_episode)
        plt.show()

    def simulate_trajectory(self, action_value_function):
        currentState = self.start
        while (currentState != self.end):
            x, y = self.convert_1D_to_2D(currentState)
            optimal_action = np.argmax(action_value_function[currentState])
            if (optimal_action == 0):
                print('N', end=" ")
            elif (optimal_action == 1):
                print('E', end=" ")
            elif (optimal_action == 2):
                print('S', end=" ")
            elif (optimal_action == 3):
                print('W', end=" ")

            next_x, next_y = self.normal_move(x, y, optimal_action)
            nextState = self.convert_2D_to_1D(next_x, next_y)
            currentState = nextState

        print("")


if __name__ == "__main__":
    gridworld = NormalWindyGridworld()
    # action_value_function = gridworld.sarsa(episodes=170, gamma=1, epsilon=0.1, alpha=0.5)
    action_value_function = gridworld.q_learning(episodes=170, gamma=1, epsilon=0.1, alpha=0.5)
    # action_value_function = gridworld.expected_sarsa(
        # episodes=170, gamma=1, epsilon=0.1, alpha=0.5)
    # action_value_function = gridworld.dyna_q(episodes=170, gamma=0.95, epsilon=0.1, alpha=0.1, n=50)
    gridworld.simulate_trajectory(action_value_function)
