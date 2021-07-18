# SnakeAI
This repository contains the code for my WnCC Summer of Code - 2021 Project.

The goal of the project is to learn Reinforcement Learning by building a RL Model that plays the popular Snake Game. I will be doing small exercises along the way whose source code can be found in this repository.

## Game Demo
I am playing the game. (fairly well if I say so myself)  
[Link to the Gameplay](https://drive.google.com/drive/folders/1kixSPSeSGwu6KX9O60KJUzhxoQp0nPv8)

## Assignment 1 - Maze Solver using Value Iteration
In this assignment, we had to implement a maze solver using Value Iteration. The program runs quickly on almost all grid sizes except the 10x10 one. I plan on writing an alternative program that uses a dictionary-based approach rather than vectorized matrix computations.

Details of the assignment can be found [here](maze-solver/README.md)

## Assignment 2 - Sutton & Barto's Windy Gridworld using Model-free control
In this assignment, we solve the Windy Gridworld problem using different Model-free approaches - SARSA(0), Q-Learning, Expected-SARSA, Dyna-Q. Dyna-Q proves to be the best, outperforming all the other algorithms while Q-Learning and Expected-SARSA have similar performances. I plan on adding an implemention of SARSA(&lambda;) and comparing it with other approaches listed above.  
[Sutton & Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)  
[Link to the Assignment 2](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/pa-2/programming-assignment-3.html)

## Simple Tabular RL Agent
I have added a simple Tabular RL agent to play the Snake Game. The game has an extremely large state space representation and we must use a reduced representation to make the task feasible.
### State Space
My representation uses 7 bits of information to describe the current state of the snake:
* 4 bits of information to define the relative position of the fruit with respect to the head of the snake
* 3 bits of information for obstacles right in front of the head, to the immediate right and left of the head

### Action Space
The snake has 3 possible actions:
* Do nothing: The snake continues to move in the same direction
* Turn right: The snake turns right to change its direction
* Turn left: The snake turns left to change its direction

### Reward Scheme
I have used a fairly simple reward scheme that can be optimized to improve the performance of the agent:
* Reward of +5 if the snake moves closer to the fruit
* Reward of -5 if the snake moves away from the fruit
* Reward of +500 for eating the fruit
* Reward of -1000 for crashing  

### Hyperparameters
The starting learning rate and &epsilon; parameter for an &epsilon;-greedy policy are 0.5 and 0.01. Without decaying these hyperparameters, the training behaviour of the agent is extremely erratic. With annealing, the performance is more consistent. The agent has achieved a maximum score of 64.

## Resources
Please find below a list of resources created by our mentor, [**Shubham Lohiya**](https://github.com/shubhlohiya) to guide us through the project.

[Learning Resources](https://www.notion.so/SOC-Snake-AI-Project-471ff57983a24f749ca0ec08df8c9472 "Learning Resources")
