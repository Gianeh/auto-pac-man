# Classes for State counting and reinforcement learning training
from PIL import Image
from math import comb as binomial
from math import prod
from pprint import pprint
from itertools import product
from numpy import argmax, random
from random import choices
from time import sleep
import pygame
from helper_functions import is_terminal, diff_norm, pacman_move, ghost_move, is_win_terminal, is_lose_terminal, ghost_move_manhattan

class State_initializer:
    def __init__(self, map_filename="dumb_map", load_from_txt=False, logging=False):
        """
        Initial state creation and map loading from an image or a txt file

        map_filename:   Name of the map file (without extension)

        laod_from_txt:  Flag to load the map from a txt file instead of an image

        logging:        Flag to enable logging of the algorithm steps
        """
        # Save the constructor parameters
        self.filename = map_filename

        if load_from_txt:
            self.map = self.create_map_from_txt(logging)
        else:
            self.map = self.create_map(Image.open("./maps/"+map_filename+".png"), logging)

        # Compute the number of free positions in the map avoiding walls
        self.possible_positions = [(x, y) for y in range(len(self.map)) for x in range(len(self.map[0])) if self.map[y][x] != 1]
        self.free_positions = len(self.possible_positions)
        if logging: print(f"Number of free positions: {self.free_positions}")
        
        self.number_of_movables = 0
        self.number_of_ghosts = 0
        self.number_of_candies = 0
        self.candies_positions = {}
        self.initial_state = self.create_initial_state(logging)
        
        self.number_of_possible_states, self.number_of_terminal_states = self.count_states(logging)

    def create_map_from_txt(self, logging):
        # Dictionary of objects in the map (txt file)
        objects = {
            " ": 0, # floor
            "w": 1, # wall
            "c": 2, # candy
            "p": 3, # pacman
            "g": 4  # ghost
        }
        map = []
        # Read and parse the txt file
        with open("./txt_maps/"+self.filename+".txt", "r") as file:
            for line in file: 
                map.append([objects[c] for c in line.strip()])
            
        if logging:
            print()
            pprint(map)
            print()

        return map

    def create_map(self, image, logging):
        # Dictionary of colors in the map (png image)
        color_codes = {
            (255, 255, 255): 0, # white - floor
            (0, 0, 0): 1, # black - wall
            (255, 0, 0): 2, # red - candy
            (0, 255, 0): 3, # green - pacman
            (0, 0, 255): 4, # blue - ghost
        }
        map = []
        # Parse the image and create the map
        for y in range(image.size[1]):
            map.append([])
            for x in range(image.size[0]):
                # Get the rgb tuple of the pixel for each position in the image
                rgb = (image.getpixel((x, y))[0], image.getpixel((x, y))[1], image.getpixel((x, y))[2])
                # Append the color code to the map
                map[y].append(color_codes[rgb])
        
        if logging:
            print()
            pprint(map)
            print()   

        return map
    
    def create_initial_state(self, logging):
        initial_state = [(0,0)] # pacman position

        # Add the ghosts and pacman
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                if self.map[y][x] == 3:
                    initial_state[0] = (x, y)
                elif self.map[y][x] == 4:
                    initial_state.append((x, y))

        # The state is now [pacman, ghost1, ghost2, ...]
        self.number_of_movables = len(initial_state)
        self.number_of_ghosts = self.number_of_movables - 1

        # Add the candies
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                if self.map[y][x] == 2:
                    initial_state.append(1)
                    self.candies_positions[len(initial_state) - 1] = (x,y) 
        
        # The state is now [pacman, ghost1, ghost2, ..., candy1, candy2, ...]
        self.number_of_candies = len(initial_state) - self.number_of_movables

        if logging: print(f"Initial state (from loaded map): {initial_state}\n")

        return initial_state
    
    def count_states(self, logging):
        number_of_states = 0
        number_of_terminal_states = 0

        # Possible states with at least one candy at 1
        for i in range(1, self.number_of_candies + 1):
            number_of_states += (self.free_positions - i) * (self.free_positions ** self.number_of_ghosts) * binomial(self.number_of_candies, i)

        # Accounting for all possible states with all candies to 0
        number_of_states += self.number_of_candies * (self.free_positions ** self.number_of_ghosts)

        # Accounting for all possible winning terminal states (all candies are 0, pacman is not eaten by a ghost and pacman is on a candy)  
        number_of_terminal_states = self.number_of_candies * ((self.free_positions - 1) ** self.number_of_ghosts)

        # Accounting for all possible losing terminal states (pacman is eaten by one or more ghosts) and at least one candy is 1
        for c in range(1, self.number_of_candies + 1):
            for i in range(1, self.number_of_ghosts + 1):
                number_of_terminal_states += (self.free_positions - c) * ((self.free_positions - 1) ** (self.number_of_ghosts - i)) * binomial(self.number_of_ghosts, i) * binomial(self.number_of_candies, c) 

        # Accounting for all candies eaten (all candies are 0) and at least one ghost is on the same position of pacman (pacman eaten)
        for i in range(1, self.number_of_ghosts + 1):
            number_of_terminal_states += self.number_of_candies * ((self.free_positions - 1) ** (self.number_of_ghosts - i)) * binomial(self.number_of_ghosts, i)

        if logging:
            print(f"\nNumber of possible states: {number_of_states}")
            print(f"Number of terminal states: {number_of_terminal_states}\n")

        return number_of_states, number_of_terminal_states
    

class Policy_iterator:
    def __init__(self, initializer, renderer=None, max_episodes=1000, alpha=1e-1, gamma=9e-1, epsilon=1e-1, lose_reward=-1e3, win_reward=5e2, move_reward=-1, eat_reward=1e1, power=10, logging=False):
        """
        Generalized Policy iteration algorithm (requires State_initializer instance)

        initializer:        State_initializer instance, built as State_initializer(map_filename, logging)

        max_episodes:       Number of episodes to run the algorithm
        
        alpha:              Learning rate, weights the new information against the old information during the update of the Q function

        gamma:              Discount factor, weights the future rewards against the current rewards

        epsilon:            Exploration rate, the higher the value the more the agent will explore the environment

        lose_reward:        reward of losing the game (eaten by a ghost)

        win_reward:         reward of winning the game (eating all candies)

        move_reward:        reward of moving from one state to another without eating a candy or being eaten by a ghost

        eat_reward:         reward of eating a candy

        power:              Power parameter for the ghost moves, the higher the power the more the ghosts will try to get closer to pacman, power = 0 means the ghosts move randomly
        
        logging:            Flag to enable logging of the algorithm steps
        """
        self.map = initializer.map
        self.initial_state = initializer.initial_state
        self.number_of_movables = initializer.number_of_movables
        self.number_of_ghosts = initializer.number_of_ghosts
        self.number_of_candies = initializer.number_of_candies
        self.possible_positions = initializer.possible_positions
        self.candies_positions = initializer.candies_positions
        self.possible_states = initializer.number_of_possible_states
        self.filename = initializer.filename
        self.power = power

        # save policy iteration hyperparameters
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # reward function parameters
        self.lose_reward = lose_reward
        self.win_reward = win_reward
        self.move_reward = move_reward
        self.eat_reward = eat_reward
                
        # possible moves in a 2D grid
        self.moves = {
            0: (0, -1), # up
            1: (0, 1),  # down
            2: (-1, 0), # left
            3: (1, 0),  # right
            4: (0, 0)   # stay
        }

        # Initialize the Q function
        self.Q = {}

        self.episodes = 1

        # Logging flag
        self.logging = logging

        # Renderer object 
        self.renderer = renderer

    def reward(self, state, eaten):
        # if pacman is eaten by a ghost
        if state[0] in state[1:self.number_of_movables]:
            return self.lose_reward
        # -- Elif pacman is adjacent (including diagonals) to a ghost, it's penalized
        for i in range(1, self.number_of_movables):
            ghost_x, ghost_y = state[i]  # Ghost

            # List of cells adjacent (orthogonally + diagonally) to Pac-Man
            neighbors = [
                (state[0][0] - 1, state[0][1]),     # left
                (state[0][0] + 1, state[0][1]),     # right
                (state[0][0], state[0][1] - 1),     # up
                (state[0][0], state[0][1] + 1),     # down
                (state[0][0] - 1, state[0][1] - 1), # top-left
                (state[0][0] - 1, state[0][1] + 1), # bottom-left
                (state[0][0] + 1, state[0][1] - 1), # top-right
                (state[0][0] + 1, state[0][1] + 1)  # bottom-right
            ]
            
            # If the ghost is in any of those neighboring cells, apply the lose reward
            if (ghost_x, ghost_y) in neighbors:
                return self.lose_reward
            
        # if pacman ate a candy
        if eaten:
            if sum(state[self.number_of_movables:]) == 0:
                return self.win_reward
            return self.eat_reward
        # otherwise
        return self.move_reward

    def pi(self, state):
        possible_moves = [move for move in self.moves if self.map[state[0][1] + self.moves[move][1]][state[0][0] + self.moves[move][0]] != 1]
        # exploration
        if random.rand() < self.epsilon:
            return random.choice(possible_moves)
        # exploitation
        action_argmax = possible_moves[0]
        for action in possible_moves:
            if self.Q.get((state, action), random.rand()) > self.Q.get((state, action_argmax), random.rand()):
                action_argmax = action

        return action_argmax

    def run(self):
        if self.logging: print(f"Running Policy Iteration algorithm for {self.max_episodes} episodes...\n")

        action = 0 # action placeholder for renderer object
        current_state = self.initial_state.copy()
        next_state = self.initial_state.copy()
        if self.logging: print(f"Episode {self.episodes}")
        while self.episodes <= self.max_episodes:

            # Render the training
            if self.renderer is not None:
                self.renderer.render(current_state, action)

            action = self.pi(tuple(current_state))
            next_state, eaten = pacman_move(current_state, self.moves[action], self.number_of_movables, self.candies_positions, self.map)
            
            # Stochastic ghost moves and their probabilities
            possible_ghosts_actions = []
            ghosts_actions_pmfs = []
            for ghost_index in range(1, self.number_of_movables):
                possible_ghost_action_list , pmf = ghost_move_manhattan(next_state, ghost_index, self.moves, self.map, self.power)
                possible_ghosts_actions.append(possible_ghost_action_list)
                ghosts_actions_pmfs.append(pmf)

            permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
            permuted_pmfs = [list(p) for p in product(*ghosts_actions_pmfs)]
            ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]


            next_states = []
            for actions in permuted_actions:
                ghosts_state = []
                for i in range(len(actions)):
                    ghosts_state.append((actions[i][0] + next_state[1+i][0], actions[i][1] + next_state[1+i][1]))
                next_states.append([next_state[0]] + ghosts_state + next_state[self.number_of_movables:])
            
            next_state = choices(next_states, weights=ghosts_action_permutations_pmfs, k=1)[0]
            reward = self.reward(next_state, eaten)

            # Q function update
            if is_terminal(next_state, self.number_of_movables):
                next_possible_moves = [4]   # stay
            else:
                next_possible_moves = [move for move in self.moves if self.map[next_state[0][1] + self.moves[move][1]][next_state[0][0] + self.moves[move][0]] != 1]

            self.Q[(tuple(current_state), action)] = self.Q.get((tuple(current_state), action), 0) + \
                self.alpha * (reward + self.gamma * max([self.Q.get((tuple(next_state), action), 0) for action in next_possible_moves]) - self.Q.get((tuple(current_state), action), 0))
            
            current_state = next_state
            print(current_state)
            if is_terminal(current_state, self.number_of_movables):
                self.episodes += 1
                if self.renderer is not None:
                    self.renderer.render(current_state, action)
                current_state = self.initial_state.copy()
                # randomize pacman position avoiding ghosts
                current_state[0] = self.possible_positions[random.choice(len(self.possible_positions))]
                while current_state[0] in current_state[1:self.number_of_movables] or current_state[0] in self.candies_positions.values():
                    current_state[0] = self.possible_positions[random.choice(len(self.possible_positions))]
                if self.logging: print(f"Episode {self.episodes}")




class Renderer:
    def __init__(self, initializer, tile_size=50, fps=10, logging=False):
        # Save the initializer parameters
        self.map = initializer.map
        self.number_of_movables = initializer.number_of_movables
        self.candies_positions = initializer.candies_positions

        # Save the game parameters
        self.tile_size = tile_size
        self.fps = fps
        
        # Initialize the screen
        self.screen = self.init_screen()

        # Clock
        self.clock = pygame.time.Clock()

        # Game images initialization
        self.pacman_image = None
        self.ghost_image = None
        self.candy_image = None
        self.floor_image = None
        self.wall_image = None
        self.init_images()

        self.pacman_images = {
            0: pygame.image.load("./images/u_pacman.png"),
            1: pygame.image.load("./images/d_pacman.png"),
            2: pygame.image.load("./images/l_pacman.png"),
            3: pygame.image.load("./images/r_pacman.png"),
            4: pygame.image.load("./images/r_pacman.png"),
        }

    def init_screen(self):
        pygame.display.set_caption("Pac-Man")
        return pygame.display.set_mode((len(self.map[0]) * self.tile_size, len(self.map) * self.tile_size))
    
    def init_images(self):
        self.pacman_image = pygame.image.load("./images/r_pacman.png")
        self.pacman_image = pygame.transform.scale(self.pacman_image, (self.tile_size, self.tile_size))

        # Ghost image for the first ghost (could be player controlled)
        self.first_ghost_image = pygame.image.load("./images/ghost_1.png")
        self.first_ghost_image = pygame.transform.scale(self.first_ghost_image, (self.tile_size, self.tile_size))
        
        # Ghost image for the other ghosts
        self.ghost_image = pygame.image.load("./images/ghost_3.png")
        self.ghost_image = pygame.transform.scale(self.ghost_image, (self.tile_size, self.tile_size))

        self.candy_image = pygame.image.load("./images/candy_1.png")
        self.candy_image = pygame.transform.scale(self.candy_image, (self.tile_size, self.tile_size))
        self.floor_image = pygame.Surface((self.tile_size, self.tile_size))
        self.floor_image.fill((0, 0, 0))
        self.wall_image = pygame.image.load("./images/wall3.png")
        self.wall_image = pygame.transform.scale(self.wall_image, (self.tile_size, self.tile_size))

    def render(self, state, action):
        # Draw walls and floors
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                if self.map[y][x] == 1:
                    self.screen.blit(self.wall_image, (x * self.tile_size, y * self.tile_size))
                else: 
                    self.screen.blit(self.floor_image, (x * self.tile_size, y * self.tile_size))

        # Draw candies            
        for candy_index in self.candies_positions:
            if state[candy_index] == 1:
                self.screen.blit(self.candy_image, (self.candies_positions[candy_index][0] * self.tile_size, self.candies_positions[candy_index][1] * self.tile_size))

        # Draw pacman 
        self.pacman_image = self.pacman_images[action]
        self.pacman_image = pygame.transform.scale(self.pacman_image, (self.tile_size, self.tile_size))

        self.screen.blit(self.pacman_image, (state[0][0] * self.tile_size, state[0][1] * self.tile_size))

        # Draw ghosts
        if self.number_of_movables > 1:
            self.screen.blit(self.first_ghost_image, (state[1][0] * self.tile_size, state[1][1] * self.tile_size))
        for ghost_index in range(2, self.number_of_movables):
            self.screen.blit(self.ghost_image, (state[ghost_index][0] * self.tile_size, state[ghost_index][1] * self.tile_size))
        
        
        pygame.display.flip()
        self.clock.tick(self.fps)

        # Event handling - listen for arrow key down and up to update the fps
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.fps += 50
                if event.key == pygame.K_DOWN:
                    self.fps -= 50
                if self.fps <= 0:
                    self.fps = 1
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()