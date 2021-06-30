import pygame
import sys
import numpy as np

from pygame.math import Vector2
from settings import Settings
from snake import Snake
from fruit import Fruit
from snake_game import SnakeGame


class Agent:
	def __init__(self):
		# state-space representation
		# bit-0 : reward ahead
		# bit-1 : reward behind
		# bit-2 : reward to the right
		# bit-3 : reward to the left
		# bit-4 : obstacle ahead
		# bit-5 : obstacle to the right
		# bit-6 : obstacle to the left
		self.states = 128

		# actions
		# 0 : do nothing
		# 1 : turn right
		# 2 : turn left
		self.actions = 3

		self.snake_game = SnakeGame()
		self.action_value_function = np.zeros((states, actions))

	def get_state(self):
		bit_0 = self.snake_game.snake.direction.x * (self.snake_game.fruit.position.x - self.snake_game.snake.body[0].x) + self.snake_game.snake.direction.y * (
		    self.snake_game.fruit.position.y - self.snake_game.snake.body[0].y) > 0

		bit_1 = self.snake_game.snake.direction.x * (self.snake_game.fruit.position.x - self.snake_game.snake.body[0].x) + self.snake_game.snake.direction.y * (
		    self.snake_game.fruit.position.y - self.snake_game.snake.body[0].y) < 0

		bit_2 = self.snake_game.snake.direction.x * (self.snake_game.fruit.position.y - self.snake_game.snake.body[0].y) + self.snake_game.snake.direction.y * (
		    self.snake_game.fruit.position.x - self.snake_game.snake.body[0].x) > 0

		bit_3 = self.snake_game.snake.direction.x * (self.snake_game.fruit.position.y - self.snake_game.snake.body[0].y) + self.snake_game.snake.direction.y * (
		    self.snake_game.fruit.position.x - self.snake_game.snake.body[0].x) < 0

		bit_4 = 0
		bit_5 = 0
		bit_6 = 0
		if self.snake_game.snake.direction.y == 0:
			block_ahead_x = self.snake_game.snake.body[0].x + self.snake_game.snake.direction.x
			block_ahead_y = self.snake_game.snake.body[0].y

			block_to_the_right_x = self.snake_game.snake.body[0].x
			block_to_the_right_y = self.snake_game.snake.body[0].y + self.snake_game.snake.direction.x 

			block_to_the_left_x = self.snake_game.snake.body[0].x
			block_to_the_left_y = self.snake_game.snake.body[0].y - self.snake_game.snake.direction.x

		elif self.snake_game.snake.direction.x == 0:
			block_ahead_x = self.snake_game.snake.body[0].x 
			block_ahead_y = self.snake_game.snake.body[0].y + self.snake_game.snake.direction.y

			block_to_the_right_x = self.snake_game.snake.body[0].x - self.snake_game.snake.direction.y 
			block_to_the_right_y = self.snake_game.snake.body[0].y 

			block_to_the_left_x = self.snake_game.snake.body[0].x + self.snake_game.snake.direction.y
			block_to_the_left_y = self.snake_game.snake.body[0].y

		for block in self.snake_game.snake.body[1:]:
			if block.x == block_ahead_x and block.y == block_ahead_y:
				bit_4 = 1

			if block.x == block_to_the_right_x and block.y == block_to_the_right_y:
				bit_5 = 1

			if block.x == block_to_the_left_x and block.y == block_to_the_left_y:
				bit_6 = 1

		if block_ahead_x == 0 or block_ahead_x == self.snake_game.settings.cell_number - 1:
			bit_4 = 1

		if block_to_the_right_y == 0 or block_to_the_right_y == self.snake_game.settings.cell_number - 1:
			bit_5 = 1

		if block_to_the_left_y == 0 or block_to_the_left_y == self.snake_game.settings.cell_number - 1:
			bit_6 = 1	

		return bit_0 + 2 * bit_1 + 4 * bit_2 + 8 * bit_3 + 16 * bit_4 + 32 * bit_5 + 64 * bit_6	
	

	def epsilon_greedy(self, state, epsilon=0.1):
		if np.random.uniform(low=0.0, high=1.0) < epsilon:
            action = np.random.randint(0, self.actions)
        else:
            action = np.random.choice(np.flatnonzero(self.action_value_function[state, :] == self.action_value_function[state, :].max()))

        return action

    def update_direction(self, action):
    	if action == 0:
    		pass

		elif action == 1:
			if self.snake_game.snake.direction.x == 1:
				self.snake_game.snake.direction = (0, 1)

			elif self.snake_game.snake.direction.x == -1:
				self.snake_game.snake.direction = (0, -1)

			elif self.snake_game.snake.direction.y == 1:
				self.snake_game.snake.direction = (-1, 0)

			elif self.snake_game.snake.direction.y == -1
				self.snake_game.snake.direction = (1, 0):

		elif action == 2:
			if self.snake_game.snake.direction.x == 1:
				self.snake_game.snake.direction = (0, -1)

			elif self.snake_game.snake.direction.x == -1:
				self.snake_game.snake.direction = (0, 1)

			elif self.snake_game.snake.direction.y == 1:
				self.snake_game.snake.direction = (1, 0)

			elif self.snake_game.snake.direction.y == -1
				self.snake_game.snake.direction = (-1, 0)

	def check_termination(self):
		# exit the game if snake goes outside of the screen
        if not 0 <= self.snake_game.snake.body[0].x < self.snake_game.settings.cell_number:
            self.snake_game.snake.reset()
            return True
        if not 0 <= self.snake_game.snake.body[0].y < self.snake_game.settings.cell_number:
            self.snake_game.snake.reset()
            return True

        # exit the game check if snake collides with itself
        for block in self.snake_game.snake.body[1:]:
            if block == self.snake_game.snake.body[0]:
                self.snake_game.snake.reset()
                return True

        return False

	def q_learning_episode(self, gamma=1, epsilon=0.1, alpha=0.5):
		# create a time based user event to move the snake and to check for collisions
        SCREEN_UPDATE = pygame.USEREVENT
        # this event is triggered every 150ms
        pygame.time.set_timer(SCREEN_UPDATE, 150)

		# Reward Scheme
		# Snake moves towards the fruit : -1
		# Snake moves away from the fruit : +1
		# Snake eats the fruit : +10
		# Snake crashes : -10
		moving_towards_the_fruit_reward = +1
		moving_away_from_the_fruit_reward = -1
		eating_the_fruit_reward = +10
		crashing_reward = -10

		while True:
			current_state = self.get_state()
			action = self.epsilon_greedy(current_state, epsilon=0.1)
			self.update_direction(action)
			reward = moving_away_from_the_fruit_reward
			if self.snake_game.snake.direction.x * (self.snake_game.fruit.position.x - self.snake_game.snake.body[0].x) + self.snake_game.snake.direction.y * (self.snake_game.fruit.position.y - self.snake_game.snake.body[0].y) > 0:
				reward = moving_towards_the_fruit_reward
			else:
				reward = moving_away_from_the_fruit_reward

			if self.snake_game.snake.new_block == True:
				reward = eating_the_fruit_reward

			self.snake_game.snake.move_snake()
			if self.check_termination() == True:
				reward = crashing_reward

			next_state = self.get_state()

			self.snake_game.check_collision()
			
			error = reward + gamma * np.max(self.action_value_function[next_state, :]) - self.action_value_function[current_state, action]
			action_value_function[current_state, action] += alpha * error

			if self.check_termination() == True:
				break

if __name__ == '__main__':
	agent = Agent()
	episodes = 170
	for episode in range(episodes):
		agent.q_learning_episode(gamma=1, epsilon=0.1, alpha=0.5)