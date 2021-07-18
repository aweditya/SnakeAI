# SnakeAI
The goal of the project is to learn Reinforcement Learning by building a RL Model that plays the popular Snake Game. We explored _Tabular-based RL Methods_ starting with _Model-based methods_ like **Dynamic Programming** which require a complete knowledge of the environment and _Model-free methods_ like **Monte-Carlo and Temporal-Difference Methods** which only require sampled returns.
I will be doing small exercises along the way whose source code can be found in this repository.

This project was done as part of WnCC's Summer of Code - 2021.

## Game Demo
Please find below a video of the basic Snake Game built using the Pygame library.
[Link to the Gameplay](https://drive.google.com/drive/folders/1kixSPSeSGwu6KX9O60KJUzhxoQp0nPv8)

## Assignment 1 - Maze Solver using Value Iteration
In this assignment, we had to implement a maze solver using **Value Iteration**, a Model-based RL approach. The program runs quickly on almost all grid sizes except the 10 x 10 one. I plan on writing an alternative program that uses a dictionary-based approach rather than vectorized matrix computations.

Details of the assignment can be found [here](maze-solver/README.md).

## Assignment 2 - Sutton & Barto's Windy Gridworld using Model-free control
In this assignment, we solve the Windy Gridworld problem using different Model-free approaches - **SARSA(0), Q-Learning, Expected-SARSA, Dyna-Q**. Dyna-Q proves to be the best, outperforming all the other algorithms while Q-Learning and Expected-SARSA have similar performances. I plan on adding an implemention of SARSA(&lambda;) and comparing it with other approaches listed above.  

Details of the assignment can be found [here](windy-gridworld/README.md).

[Sutton & Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)  


## Simple Tabular RL Agent
I have added a simple Tabular RL agent to play the Snake Game. The game has an extremely large state space representation and we must use a reduced representation to make the task feasible.
This has its shortcomings since the agent develops a tendency to coil and crash into itself. A need for a better representation and reward scheme emerges. However, the agent still manages to achieve a maximum score of 64 during training.

Further details can be found [here](main-game/)

## Future Prospects
This project was a starting point for my study of Reinforcement Learning Methods. We focused solely on _Tabular-based RL Methods_ which are impractical in real-life settings. In such cases, we resort to _Value Function Approximators_ including Linear & Non-linear function approximators. I plan on adding code for a _Deep Learning_ method using _Convolutional Neural Networks_. This approach is a better model of how humans learn-and-play games since we do not have direct access to the games internal state variables.

## Resources
Please find below a list of resources created by our mentor, [**Shubham Lohiya**](https://github.com/shubhlohiya) to guide us through the project.

[Learning Resources](https://www.notion.so/SOC-Snake-AI-Project-471ff57983a24f749ca0ec08df8c9472 "Learning Resources")
