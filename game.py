import pygame
import random
import sys

from pygame.math import Vector2

# SNAKE class representing the agent in the RL model
class SNAKE:
    def __init__(self):
        # the snake has a starting length of 3 cells
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        # the snake moves to the right initially
        self.direction = Vector2(1,0)
        self.new_block = False

        # load all the snake blocks to be displayed later
        self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()

        # load the sound to be played when the snake gets a reward
        self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')

    # find the direction in which the head is pointing
    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0]
        if head_relation == Vector2(1,0):
            self.head = self.head_left

        elif head_relation == Vector2(-1,0):
            self.head = self.head_right

        elif head_relation == Vector2(0,1):
            self.head = self.head_up

        elif head_relation == Vector2(0,-1):
            self.head = self.head_down

    # find the direction in which the tail is pointing
    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1,0):
            self.tail = self.tail_left

        elif tail_relation == Vector2(-1,0):
            self.tail = self.tail_right

        elif tail_relation == Vector2(0,1):
            self.tail = self.tail_up

        elif tail_relation == Vector2(0,-1):
            self.tail = self.tail_down

    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()

        for index, block in enumerate(self.body):
            # create a rect for postioning
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)

            # draw the head of the snake
            if index == 0:
                screen.blit(self.head, block_rect)

            # draw the tail of the snake
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)

            # draw the remaining blocks of the snake
            else:
                # check the relative positioning of the next and previous block
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block

                # the blocks are on the same vertical line
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)

                # the blocks are on the same horizontal line
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)

                # handling blocks at a turn
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        screen.blit(self.body_tl, block_rect)

                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        screen.blit(self.body_bl, block_rect)

                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        screen.blit(self.body_tr, block_rect)

                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        screen.blit(self.body_br, block_rect)

    def move_snake(self):
        # if the snake ate an apple, increase it's length by one and update
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def play_crunch_sound(self):
        self.crunch_sound.play()

    def reset(self):
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        self.direction = Vector2(0,0)

# fruit class representing the reward
class FRUIT:
    def __init__(self):
        self.randomize()

    def randomize(self):
        # specify a random starting x and y position
        self.x = random.randint(0, cell_number - 1)
        self.y = random.randint(0, cell_number - 1)

        # store position as a vector
        self.position = Vector2(self.x, self.y)

    def draw_fruit(self):
        x_pos = int(self.position.x * cell_size)
        y_pos = int(self.position.y * cell_size)
        fruit_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)

        # draw the apple
        screen.blit(apple, fruit_rect)

# main game control
class MAIN:
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()

    def draw_grass(self):
        grass_color = (167,209,61)

        for row in range(cell_number):
            if row % 2 == 0:
                for column in range(cell_number):
                    if column % 2 == 0:
                        grass_rect = pygame.Rect(column * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
            else:
                for column in range(cell_number):
                    if column % 2 == 1:
                        grass_rect = pygame.Rect(column * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (56, 74, 12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        apple_rect = apple.get_rect(midright = (score_rect.left, score_rect.centery))
        scoreboard_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6, apple_rect.height)

        pygame.draw.rect(screen, (167, 209, 61), scoreboard_rect)
        screen.blit(score_surface, score_rect)
        screen.blit(apple, apple_rect)
        pygame.draw.rect(screen, (56, 74, 12), scoreboard_rect, 2)

    def draw_elements(self):
        self.draw_grass()
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        self.draw_score()

    def check_collision(self):
        # check if the snake has eaten an apple
        if self.fruit.position == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()
            self.snake.play_crunch_sound()

        # change the apple position in case it lands on the snake body
        for block in self.snake.body[1:]:
            if block == self.fruit.position:
                self.fruit.randomize()

    def check_fail(self):
        # exit the game if snake goes outside of the screen
        if not 0 <= self.snake.body[0].x < cell_number:
            self.game_over()
        if not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()

        # exit the game check if snake collides with itself
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()

    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()

    def game_over(self):
        pygame.quit()
        sys.exit()

pygame.mixer.pre_init(44100, -16, 2, 512) # removing lag in sound when snake eats the apple
pygame.init() # import all the modules

# game window settings
cell_size = 30
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
pygame.display.set_caption('The Snake Game')

# add a clock to set a maximum fps in the game loop
clock = pygame.time.Clock()

# load the apple image
apple = pygame.image.load('Graphics/apple.png').convert_alpha()

# load the game font to display the score
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

# create a time based user event to move the snake and to check for collisions
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150) # this event is triggered every 150ms

main_game = MAIN()
# main game loop
while True:
    for event in pygame.event.get():
        # exit the game if user presses the exit button
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == SCREEN_UPDATE:
            main_game.update()

        # handle user key events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if main_game.snake.direction.y != 1:
                    main_game.snake.direction = Vector2(0, -1)

            if event.key == pygame.K_DOWN:
                if main_game.snake.direction.y != -1:
                    main_game.snake.direction = Vector2(0, 1)

            if event.key == pygame.K_LEFT:
                if main_game.snake.direction.x != 1:
                    main_game.snake.direction = Vector2(-1, 0)

            if event.key == pygame.K_RIGHT:
                if main_game.snake.direction.x != -1:
                    main_game.snake.direction = Vector2(1, 0)

    # color the screen RGB = (175, 210, 70)
    screen.fill((175, 210, 70))
    main_game.draw_elements()

    # update the screen
    pygame.display.update()

    clock.tick(60) # set the maximum fps = 60
