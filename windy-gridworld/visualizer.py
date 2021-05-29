#! /usr/bin/python
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()


class Visualizer:
    def __init__(self, moves, solution):
        self.rows = 7
        self.cols = 10
        self.start = 3 * self.cols + 0  # (3, 0)
        self.end = 3 * self.cols + 7  # (3, 7)

        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        if (moves == "normal"):
            self.make_normal_windy_gridworld(solution)
        else:
            self.make_king_windy_gridworld(solution)

    def convert_2D_to_1D(self, x, y):
        return x * self.cols + y

    def convert_1D_to_2D(self, state):
        return state // self.cols, state % self.cols

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

    def make_normal_windy_gridworld(self, solution):
        data = open(solution, 'r')

        trajectory = []

        for line in data:
            moves = line.split(' ')
            for move in moves:
                move = move.rstrip("\n")
                if move == 'N':
                    trajectory.append(0)
                elif move == 'E':
                    trajectory.append(1)
                elif move == 'S':
                    trajectory.append(2)
                elif move == 'W':
                    trajectory.append(3)

        figure = plt.figure(figsize=(9,6))
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        plt.title('Optimal Solution using Normal Moves')

        plt.xlim(0,10)
        plt.ylim(7,0)

        grid_xticks = np.arange(0, 10, 1)
        grid_yticks = np.arange(0, 7, 1)

        ax.set_xticks(grid_xticks)
        ax.set_yticks(grid_yticks)

        plt.grid(which='both')

        current_state = self.start
        for move in trajectory:
            current_x, current_y = self.convert_1D_to_2D(current_state)
            next_x, next_y = self.normal_move(current_x, current_y, move)
            plt.plot([next_y + 0.5, current_y + 0.5], [next_x + 0.5, current_x + 0.5])

            current_state = self.convert_2D_to_1D(next_x, next_y)

        plt.tight_layout()
        plt.show()

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

        next_x = max(next_x - self.wind_strength[y], 0)
        return next_x, next_y

    def make_king_windy_gridworld(self, solution):
        data = open(solution, 'r')

        trajectory = []

        # 0-N, 1-NE, 2-E, 3-SE, 4-S, 5-SW, 6-W, 7-NW
        for line in data:
            moves = line.split(' ')
            for move in moves:
                move = move.rstrip("\n")
                if move == 'N':
                    trajectory.append(0)
                elif move == 'NE':
                    trajectory.append(1)
                elif move == 'E':
                    trajectory.append(2)
                elif move == 'SE':
                    trajectory.append(3)
                elif move == 'S':
                    trajectory.append(4)
                elif move == 'SW':
                    trajectory.append(5)
                elif move == 'W':
                    trajectory.append(6)
                elif move == 'NW':
                    trajectory.append(7)

        figure = plt.figure(figsize=(9,6))
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        plt.title('Optimal Solution using King Moves')

        plt.xlim(0,10)
        plt.ylim(7,0)

        grid_xticks = np.arange(0, 10, 1)
        grid_yticks = np.arange(0, 7, 1)

        ax.set_xticks(grid_xticks)
        ax.set_yticks(grid_yticks)

        plt.grid(which='both')

        current_state = self.start
        for move in trajectory:
            current_x, current_y = self.convert_1D_to_2D(current_state)
            next_x, next_y = self.king_move(current_x, current_y, move)
            plt.plot([next_y + 0.5, current_y + 0.5], [next_x + 0.5, current_x + 0.5])

            current_state = self.convert_2D_to_1D(next_x, next_y)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser.add_argument("--moves", type=str, default="normal")
    parser.add_argument("--solution", type=str)

    args = parser.parse_args()
    if not (args.moves == "normal" or args.moves == "king"):
        print("Only normal and king moves are supported")
        sys.exit(0)

    visualizer = Visualizer(args.moves, args.solution)