# Maze Solver
This assignment was a part of the course _CS747: Foundations of Intelligent and Learning Agents_ offered at IIT Bombay.

The task was to write code to compute an optimal policy for a given MDP using **Value Iteration, Policy Iteration and Linear Programming**. So far, I have only added the implemention of Value Iteration and plan on adding code for Policy Iteration and Linear Programming in the future.

Having implemented the algorithms, we used the solver to find the shortest path in a maze.

## Task 1: MDP Solver
I have created a python file `planner.py` which accepts the following command-line arguments:
* `--mdp` followed by the path to the MDP (examples present [here](data/mdp))
* `--algorithm` followed by one of **vi, hpi, lp**

Examples of usage (invocation from the same directory):
```bash
  python planner.py --mdp /data/mdp-4.txt --algorithm vi
```

## Task 2: Maze Solver
The first step was to encode the maze as an MDP following which, we could use `planner.py` to find an optimal policy.

Each empty tile denotes a possible state and any valid action takes you from one tile to the next. Invalid actions leave the state unchanged, for eg: a right move doesn't change the state if there is a wall to the right. To incentivize the agent to complete the task as soon as possible, all transitions incur a reward of **-1**.

We can finally simulate the optimal policy resulting in the shortest path from start to end.

The python file `encoder.py` accepts the command-line argument `--grid` followed by the path to the maze (examples present [here](data/maze)) and prints the encoded MDP.

The python file `decoder.py` accepts the following command-line arguments:
* `--grid` followed by the path to the maze
* `--value_policy` followed by the optimal policy

and prints the shortest path.

Examples of usage:
```bash
python encoder.py --grid gridfile > mdpfile
python planner.py --mdp mdpfile --algorithm vi > value_and_policy_file
python decoder.py --grid gridfile --value_policy value_and_policy_file > pathfile
```

The maze can be visualized using:
```bash
python visualize.py gridfile
```

The shortest path can be visualized using:
```bash
python visualize.py gridfile pathfile
```

[Reference to Original Assignment](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs747-a2020/pa-2/programming-assignment-2.html)
