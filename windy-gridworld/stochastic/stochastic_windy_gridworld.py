#! /usr/bin/python
import sys
import matplotlib.pyplot as plt
import numpy as np
import random


class StochasticWindyGridworld:
    def __init__(self):
        self.rows = 7
        self.cols = 10
        self.start = 3 * self.cols + 0  # (3, 0)
        self.end = 3 * self.cols + 7  # (3, 7)

        # how far is the agent pushed upwards on making a move
        self.mean_wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

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

    def king_move(self, x, y, action):
        # 0-N, 1-NE, 2-E, 3-SE, 4-S, 5-SW, 6-W, 7-NW
        if (action == 0):
            next_x = max(x - 1, 0)
            next_y = y

        elif (action == 1):
            next_x = max(x - 1, 0)
            next_y = min(y + 1, self.cols - 1)

        elif (action == 2):
            next_x = x
            next_y = min(y + 1, self.cols - 1)

        elif (action == 3):
            next_x = min(x + 1, self.rows - 1)
            next_y = min(y + 1, self.cols - 1)

        elif (action == 4):
            next_x = min(x + 1, self.rows - 1)
            next_y = y

        elif (action == 5):
            next_x = min(x + 1, self.rows - 1)
            next_y = max(y - 1, 0)

        elif (action == 6):
            next_x = x
            next_y = max(y - 1, 0)

        elif (action == 7):
            next_x = max(x - 1, 0)
            next_y = max(y - 1, 0)

        probability = np.random.uniform(low=0.0, high=1.0)
        if probability > 2 / 3:
            next_x = max(next_x - self.mean_wind_strength[y] - 2, 0)

        elif probability > 1 / 3:
            next_x = max(next_x - self.mean_wind_strength[y] - 1, 0)

        else:
            next_x = max(next_x - self.mean_wind_strength[y], 0)

        return next_x, next_y

    def sarsa(self, episodes=170, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 8

        data = []

        action_value_function = np.zeros((numStates, numActions))
        for episode in range(episodes):
            currentState = self.start
            action = self.epsilon_greedy(currentState, action_value_function,
                                         epsilon, numActions)
            while (currentState != self.end):
                data.append(episode)
                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.king_move(x, y, action)

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

        plt.title('Stochastic Windy Gridworld using SARSA(0)')
        self.plot_learning_curve(data)

    def q_learning(self, episodes=170, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 8

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
                next_x, next_y = self.king_move(x, y, action)

                nextState = self.convert_2D_to_1D(next_x, next_y)

                error = reward + gamma * \
                    np.max(action_value_function[nextState, :]) - \
                        action_value_function[currentState, action]
                action_value_function[currentState, action] += alpha * error
                currentState = nextState

        plt.title('Stochastic Windy Gridworld using Q-learning')
        self.plot_learning_curve(data)

    def plot_learning_curve(self, data):
        plt.xlabel('Time Steps')
        plt.ylabel('Episodes')
        plt.plot(data)

        plt.show()

    def dyna_q(self, episodes=50, gamma=0.95, epsilon=0.1, alpha=0.1, n=50):
        reward = -1
        numStates = self.rows * self.cols
        numActions = 8

        steps_per_episode = []

        action_value_function = np.zeros((numStates, numActions))
        model = np.zeros((numStates, numActions, 2))
        previously_observed_state = np.zeros((numStates, 1))
        previously_observed_action = np.zeros((numStates, numActions))

        for episode in range(episodes):
            currentState = self.start
            steps = 0
            while (currentState != self.end):
                print(steps)
                steps += 1
                previously_observed_state[currentState, 0] = 1
                action = self.epsilon_greedy(currentState,
                                             action_value_function, epsilon,
                                             numActions)
                previously_observed_action[currentState, action] = 1

                x, y = self.convert_1D_to_2D(currentState)
                next_x, next_y = self.king_move(x, y, action)

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

        self.plot_steps_per_episode(steps_per_episode)

    def plot_steps_per_episode(self, steps_per_episode):
        plt.xlabel('Episodes')
        plt.ylabel('Steps per episode')
        plt.plot(steps_per_episode)
        plt.show()

if __name__ == "__main__":
    gridworld = StochasticWindyGridworld()
    # gridworld.sarsa(episodes=170, gamma=1, epsilon=0.1, alpha=0.5)
    gridworld.q_learning(episodes=170, gamma=1, epsilon=0.1, alpha=0.5)
    # gridworld.dyna_q(episodes=50, gamma=0.95, epsilon=0.1, alpha=0.1, n=50)
