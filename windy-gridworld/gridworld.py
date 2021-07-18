#! /usr/bin/python
import numpy as np
import random

class WindyGridworld:
    def __init__(self, moves, stochastic):
        self.moves = moves
        self.stochastic = stochastic

        self.rows = 7
        self.cols = 10
        self.states = self.rows * self.cols

        self.start = 3 * self.cols + 0  # (3, 0)
        self.end = 3 * self.cols + 7  # (3, 7)

        if self.moves == "normal":
            self.actions = 4
        
        else:
            self.actions = 8
        
        # how far is the agent pushed upwards on making a move
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

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

        if self.stochastic and self.wind_strength[y] != 0:
            probability = np.random.uniform(low=0.0, high=1.0)
            if probability > 2 / 3:
                next_x = max(next_x - self.wind_strength[y] - 2, 0)
            elif probability > 1 / 3:
                next_x = max(next_x - self.wind_strength[y] - 1, 0)
            else:
                next_x = max(next_x - self.wind_strength[y], 0)
        else:
            next_x = max(next_x - self.wind_strength[y], 0)

        return next_x, next_y

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

        if self.stochastic and self.wind_strength[y] != 0:
            probability = np.random.uniform(low=0.0, high=1.0)
            if probability > 2 / 3:
                next_x = max(next_x - self.wind_strength[y] - 2, 0)
            elif probability > 1 / 3:
                next_x = max(next_x - self.wind_strength[y] - 1, 0)
            else:
                next_x = max(next_x - self.wind_strength[y], 0)
        else:
            next_x = max(next_x - self.wind_strength[y], 0)

        return next_x, next_y

    def move(self, x, y, action):
        next_x = next_y = 0
        if self.moves == "normal":
            next_x, next_y = self.normal_move(x, y, action)
        else:
            next_x, next_y = self.king_move(x, y, action)

        return next_x, next_y