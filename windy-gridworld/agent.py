#! /usr/bin/python
import numpy as np
import random
from gridworld import WindyGridworld

class Agent:
    def __init__(self, moves="normal", stochastic=False):
        self.gridworld = WindyGridworld(moves=moves, stochastic=stochastic)
        self.action_value_function = np.zeros((self.gridworld.states, self.gridworld.actions))

    def convert_2D_to_1D(self, cols, x, y):
        return x * cols + y

    def convert_1D_to_2D(self, cols, state):
        return state // cols, state % cols

    def epsilon_greedy(self, state, action_value_function, epsilon, actions):
        if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action = np.random.randint(0, actions)
        else:
            action = np.random.choice(np.flatnonzero(action_value_function[state, :] == action_value_function[state, :].max()))

        return action

    def sarsa_episode(self, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        episode_time = 0

        currentState = self.gridworld.start
        while currentState != self.gridworld.end:
            action = self.epsilon_greedy(currentState, self.action_value_function, epsilon, self.gridworld.actions)
            x, y = self.convert_1D_to_2D(cols=self.gridworld.cols, state=currentState)
            next_x, next_y = self.gridworld.move(x, y, action)

            nextState = self.convert_2D_to_1D(cols=self.gridworld.cols, x=next_x, y=next_y)
            nextAction = self.epsilon_greedy(nextState, self.action_value_function, epsilon, self.gridworld.actions)

            error = reward + gamma * self.action_value_function[nextState, nextAction] - self.action_value_function[currentState, action]
            self.action_value_function[currentState, action] += alpha * error
            currentState = nextState
            action = nextAction
            episode_time += 1

        return episode_time

    def q_learning_episode(self, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        episode_time = 0

        currentState = self.gridworld.start
        while currentState != self.gridworld.end:
            action = self.epsilon_greedy(currentState, self.action_value_function, epsilon, self.gridworld.actions)
            x, y = self.convert_1D_to_2D(cols=self.gridworld.cols, state=currentState)
            next_x, next_y = self.gridworld.move(x, y, action)

            nextState = self.convert_2D_to_1D(cols=self.gridworld.cols, x=next_x, y=next_y)

            error = reward + gamma * np.max(self.action_value_function[nextState, :]) - self.action_value_function[currentState, action]
            self.action_value_function[currentState, action] += alpha * error
            currentState = nextState
            episode_time += 1

        return episode_time

    def current_policy(self, state, action_value_function, epsilon, actions):
        policy = np.ones((1, actions)) * epsilon / actions
        greedy_action = np.random.choice(np.flatnonzero(action_value_function[state, :] == action_value_function[state, :].max()))
        policy[0, greedy_action] += 1 - epsilon

        return policy

    def expected_sarsa_episode(self, gamma=1, epsilon=0.1, alpha=0.5):
        reward = -1
        episode_time = 0

        currentState = self.gridworld.start
        while currentState != self.gridworld.end:            
            action = self.epsilon_greedy(currentState, self.action_value_function, epsilon, self.gridworld.actions)
            x, y = self.convert_1D_to_2D(cols=self.gridworld.cols, state=currentState)
            next_x, next_y = self.gridworld.move(x, y, action)

            nextState = self.convert_2D_to_1D(cols=self.gridworld.cols, x=next_x, y=next_y)

            error = reward + gamma * np.sum(self.current_policy(nextState, self.action_value_function, epsilon, self.gridworld.actions) * self.action_value_function[nextState, :]) - self.action_value_function[currentState, action]
            self.action_value_function[currentState, action] += alpha * error
            currentState = nextState
            episode_time += 1

        return episode_time

    def dyna_q_episode(self, gamma=0.95, epsilon=0.1, alpha=0.1, n=50):
        reward = -1
        episode_time = 0

        model = np.zeros((self.gridworld.states, self.gridworld.actions, 2))
        previously_observed_state = np.zeros((self.gridworld.states, 1))
        previously_observed_action = np.zeros((self.gridworld.states, self.gridworld.actions))

        currentState = self.gridworld.start
        while currentState != self.gridworld.end:
            previously_observed_state[currentState, 0] = 1
            action = self.epsilon_greedy(currentState, self.action_value_function, epsilon, self.gridworld.actions)
            previously_observed_action[currentState, action] = 1

            x, y = self.convert_1D_to_2D(cols=self.gridworld.cols, state=currentState)
            next_x, next_y = self.gridworld.move(x, y, action)

            nextState = self.convert_2D_to_1D(cols=self.gridworld.cols, x=next_x, y=next_y)

            error = reward + gamma * np.max(self.action_value_function[nextState, :]) - self.action_value_function[currentState, action]
            self.action_value_function[currentState, action] += alpha * error
            model[currentState, action, 0] = reward
            model[currentState, action, 1] = nextState

            currentState = nextState
            episode_time += 1

            for i in range(n):
                random_state = int(np.random.choice(np.flatnonzero(previously_observed_state[:, 0] == 1)))
                random_action = int(np.random.choice(np.flatnonzero(previously_observed_action[random_state, :] == 1)))
                reward = model[random_state, random_action, 0]
                next_state = int(model[random_state, random_action, 1])
                error = reward + gamma * np.max(self.action_value_function[next_state, :]) - self.action_value_function[random_state, random_action]
                self.action_value_function[random_state, random_action] += alpha * error

        return episode_time

    def train(self, algorithm):
        episodes = 200

        if algorithm == "sarsa":
            algorithm = self.sarsa_episode
        elif algorithm == "q-learning":
            algorithm = self.q_learning_episode
        elif algorithm == "expected-sarsa":
            algorithm = self.expected_sarsa_episode
        elif algorithm == "dyna-q":
            algorithm = self.dyna_q_episode

        progress = []    
        for episode in range(episodes):
            progress.append(algorithm())

        return np.cumsum(progress)

    def simulate_trajectory(self, action_value_function):
        currentState = self.gridworld.start
        while currentState != self.gridworld.end:
            x, y = self.convert_1D_to_2D(cols=self.gridworld.cols, state=currentState)
            optimal_action = np.argmax(self.action_value_function[currentState])
            if (optimal_action == 0):
                print('N', end=" ")
            elif (optimal_action == 1):
                print('E', end=" ")
            elif (optimal_action == 2):
                print('S', end=" ")
            elif (optimal_action == 3):
                print('W', end=" ")

            next_x, next_y = self.gridworld.move(x, y, optimal_action)
            nextState = self.convert_2D_to_1D(cols=self.gridworld.cols, x=next_x, y=next_y)
            currentState = nextState

        print()    