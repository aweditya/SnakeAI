#! /usr/bin/python
import argparse
import numpy as np

parser = argparse.ArgumentParser()


class MazeMDPSolver:
    def __init__(self, grid, value_policy):
        self.numStates = 0
        self.numActions = 4
        self.start = 0
        self.end = 0

        maze, coordinate_to_state_mapping = self.process_grid(grid)
        optimal_action = self.process_value_policy(value_policy)
        self.simulate_optimal_policy(maze, coordinate_to_state_mapping,
                                     optimal_action)

    def process_grid(self, grid):
        data = open(grid, 'r').read()
        rows = len(data.split('\n')) - 1
        cols = len(data.split()) // rows

        maze = np.zeros((rows, cols))
        for i in range(len(data.split())):
            maze[i // cols][i % cols] = data.split()[i]

        self.numStates = data.count('0') + 2
        coordinate_to_state_mapping = np.zeros((self.numStates, 2))

        currentState = 0
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                value = maze[x][y]
                if value == 0:
                    coordinate_to_state_mapping[currentState][0] = x
                    coordinate_to_state_mapping[currentState][1] = y

                    currentState += 1

                elif value == 2:
                    coordinate_to_state_mapping[currentState][0] = x
                    coordinate_to_state_mapping[currentState][1] = y

                    self.start = currentState

                    currentState += 1

                elif value == 3:
                    coordinate_to_state_mapping[currentState][0] = x
                    coordinate_to_state_mapping[currentState][1] = y

                    self.end = currentState

                    currentState += 1

        return maze, coordinate_to_state_mapping

    def process_value_policy(self, value_policy):
        data = open(value_policy, 'r')

        optimal_action = np.zeros((self.numStates, 1))
        currentState = 0
        for line in data:
            elements = line.split(' ')
            optimal_action[currentState, 0] = int(elements[1])
            currentState += 1

        return optimal_action

    def simulate_optimal_policy(self, maze, coordinate_to_state_mapping,
                                optimal_action):
        currentState = self.start
        while currentState != self.end:
            x = coordinate_to_state_mapping[currentState][0]
            y = coordinate_to_state_mapping[currentState][1]
            if optimal_action[currentState][0] == 0:
                print('N', end=" ")
                for i in range(self.numStates):
                    if coordinate_to_state_mapping[i][
                            0] == x - 1 and coordinate_to_state_mapping[i][
                                1] == y:
                        currentState = i

            elif optimal_action[currentState][0] == 1:
                print('S', end=" ")
                for i in range(self.numStates):
                    if coordinate_to_state_mapping[i][
                            0] == x + 1 and coordinate_to_state_mapping[i][
                                1] == y:
                        currentState = i

            elif optimal_action[currentState][0] == 2:
                print('E', end=" ")
                for i in range(self.numStates):
                    if coordinate_to_state_mapping[i][
                            0] == x and coordinate_to_state_mapping[i][
                                1] == y + 1:
                        currentState = i

            else:
                print('W', end=" ")
                for i in range(self.numStates):
                    if coordinate_to_state_mapping[i][
                            0] == x and coordinate_to_state_mapping[i][
                                1] == y - 1:
                        currentState = i

        print("")


if __name__ == "__main__":
    parser.add_argument("--grid", type=str)
    parser.add_argument("--value_policy", type=str)
    args = parser.parse_args()

    solver = MazeMDPSolver(args.grid, args.value_policy)
