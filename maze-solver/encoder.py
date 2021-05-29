#! /usr/bin/python
import numpy as np
import argparse
parser = argparse.ArgumentParser()


class MazeMDP:
    def __init__(self, grid):
        self.numStates = 0
        self.numActions = 4
        self.start = 0
        self.end = 0

        maze, coordinate_to_state_mapping = self.process_input(grid)
        self.generateMDP(maze, coordinate_to_state_mapping)

    def process_input(self, grid):
        data = open(grid, 'r').read()
        self.rows = len(data.split('\n')) - 1
        self.cols = len(data.split()) // self.rows

        maze = np.zeros((self.rows, self.cols))
        for i in range(len(data.split())):
            maze[i // self.cols][i % self.cols] = data.split()[i]

        self.numStates = data.count('0') + 2
        print("numStates", self.numStates)
        print("numActions", self.numActions)

        coordinate_to_state_mapping = np.zeros((self.numStates, 2))

        currentState = 0
        for x in range(1, self.rows - 1):
            for y in range(1, self.cols - 1):
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

        print("start", self.start)
        print("end", self.end)

        return maze, coordinate_to_state_mapping

    def generateMDP(self, maze, coordinate_to_state_mapping):
        reward = -1.0
        probability = 1.0
        for state_index in range(self.numStates):
            x = int(coordinate_to_state_mapping[state_index][0])
            y = int(coordinate_to_state_mapping[state_index][1])
            if maze[x][y] == 3:
                print("transition", state_index, "0", state_index, "0",
                      probability)
                print("transition", state_index, "1", state_index, "0",
                      probability)
                print("transition", state_index, "2", state_index, "0",
                      probability)
                print("transition", state_index, "3", state_index, "0",
                      probability)

            else:
                if maze[x - 1][y] == 1:
                    print("transition", state_index, "0", state_index, reward,
                          probability)
                else:
                    for i in range(self.numStates):
                        if coordinate_to_state_mapping[i][
                                0] == x - 1 and coordinate_to_state_mapping[i][
                                    1] == y:
                            up = i
                    print("transition", state_index, "0", up, reward,
                          probability)

                if maze[x + 1][y] == 1:
                    print("transition", state_index, "1", state_index, reward,
                          probability)
                else:
                    for i in range(self.numStates):
                        if coordinate_to_state_mapping[i][
                                0] == x + 1 and coordinate_to_state_mapping[i][
                                    1] == y:
                            down = i
                    print("transition", state_index, "1", down, reward,
                          probability)

                if maze[x][y + 1] == 1:
                    print("transition", state_index, "2", state_index, reward,
                          probability)
                else:
                    for i in range(self.numStates):
                        if coordinate_to_state_mapping[i][
                                0] == x and coordinate_to_state_mapping[i][
                                    1] == y + 1:
                            right = i
                    print("transition", state_index, "2", right, reward,
                          probability)

                if maze[x][y - 1] == 1:
                    print("transition", state_index, "3", state_index, reward,
                          probability)
                else:
                    for i in range(self.numStates):
                        if coordinate_to_state_mapping[i][
                                0] == x and coordinate_to_state_mapping[i][
                                    1] == y - 1:
                            left = i
                    print("transition", state_index, "3", left, reward,
                          probability)

        print("mdptype episodic")
        print("discount  1")


if __name__ == "__main__":
    parser.add_argument("--grid", type=str)
    args = parser.parse_args()

    mdp = MazeMDP(args.grid)
