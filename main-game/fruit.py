import pygame
import random
import sys

from pygame.math import Vector2

from settings import Settings



class Fruit:
    def __init__(self, screen):
        self.settings = Settings()
        self.screen = screen

        # load the apple image
        self.apple = pygame.image.load('Graphics/apple.png').convert_alpha()

        self.randomize()

    def randomize(self):
        # specify a random starting x and y position
        self.x = random.randint(0, self.settings.cell_number - 1)
        self.y = random.randint(0, self.settings.cell_number - 1)

        # store position as a vector
        self.position = Vector2(self.x, self.y)

    def draw_fruit(self):
        x_pos = int(self.position.x * self.settings.cell_size)
        y_pos = int(self.position.y * self.settings.cell_size)
        fruit_rect = pygame.Rect(
            x_pos, y_pos, self.settings.cell_size, self.settings.cell_size)

        # draw the apple
        self.screen.blit(self.apple, fruit_rect)
