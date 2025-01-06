# Classes for Value iteration algorithm and states enumeration
from PIL import Image
from math import comb as binomial
from math import prod
from pprint import pprint
from sys import getsizeof
from itertools import product
from numpy import argmin, random
from time import time, sleep
import pygame
from helper_functions import is_terminal, diff_norm, pacman_move, ghost_move, is_win_terminal, is_lose_terminal, ghost_move_manhattan

class States_enumerator:
    def __init__(self, map_filename="dumb_map", load_from_txt=False, logging=False):
        """
        State enumeration algorithm from a map loaded from an image or a txt file

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
        self.states = self.enumerate_states(logging)
        if logging: print(f"Size of the states list: {getsizeof(self.states)} bytes\n")


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
    
    def enumerate_states(self, logging):
        states = []
        terminal_states_count = 0
        # Add pacman permutations
        for pacman_position in self.possible_positions:
            new_state = [pacman_position] + self.initial_state[1:]

            # Candy eaten check
            for candy_index in self.candies_positions:
                # If pacman is on a candy the candy must be 0
                if pacman_position == self.candies_positions[candy_index]:
                    new_state[candy_index] = 0
            
            if is_terminal(new_state, self.number_of_movables):
                terminal_states_count += 1
            states.append(new_state)

        if logging: print(f"\nNumber of states after pacman permutations:\n-\tGeneric states: {len(states)}\n-\tTerminal states: {terminal_states_count}")

        # Add ghosts permutations
        for ghost_index in range(1, self.number_of_movables):
            for s in range(len(states)):
                for ghost_position in self.possible_positions:
                    if ghost_position == self.initial_state[ghost_index]:
                        continue        # such state already exists
                    new_state = states[s].copy()
                    new_state[ghost_index] = ghost_position

                    if is_terminal(new_state, self.number_of_movables):
                        terminal_states_count += 1
                    states.append(new_state)

        if logging: print(f"\nNumber of states after ghosts permutations:\n-\tGeneric states: {len(states)}\n-\tTerminal states: {terminal_states_count}")

        # Add candies permutations
        for candy_index in self.candies_positions:
            for s in range(len(states)):
                if states[s][candy_index] == 0:
                    continue        # such state already exists
                new_state = states[s].copy()
                new_state[candy_index] = 0

                # Check if a state with all candies 0 is possible (pacman on a candy)
                if sum(new_state[self.number_of_movables:]) == 0:
                    if new_state[0] in self.candies_positions.values():
                        terminal_states_count += 1
                        states.append(new_state)
                # Otherwise add the state with at least one candy not eaten
                else:
                    if is_terminal(new_state, self.number_of_movables):
                        terminal_states_count += 1
                    states.append(new_state)

        if logging:
            print(f"\nNumber of states after candies permutations:\n-\tGeneric states: {len(states)}\n-\tTerminal states: {terminal_states_count}\n")

            if len(states) == self.number_of_possible_states and terminal_states_count == self.number_of_terminal_states:
                print("Calculated number of possible states is empirically correct")
            else:
                print("Calculated number of possible states is empirically incorrect")
                print(f"Computed theoretical number of states={self.number_of_possible_states} - Enumerated number of states={len(states)}")
                print(f"Computed theoretical number of terminal states={self.number_of_terminal_states} - Enumerated number of terminal states={terminal_states_count}")

        return states

class Value_iterator:
    def __init__(self, enumerator, alpha=9e-1, delta=1e-1, epsilon=1, lose_cost=1e3, win_cost=-5e2, move_cost=1, eat_cost=-1e1, power=0, control_accuracy=False, logging=False):
        """
        Value itaration algorithm from enumerated states (requires States_enumerator instance)

        enumerator:     States_enumerator instance, built as States_enumerator(map_filename, logging)

        alpha:          Discount factor, weights the importance of previously calculated Value function in policy update

        delta:          Convergence threshold, maximum value of absolute difference between two iterations of the Value function

        epsilon:        Accuracy control parameter, used to control the accuracy of the value function from the optimum

        lose_cost:      Cost of losing the game (eaten by a ghost)

        win_cost:       Cost of winning the game (eating all candies)

        move_cost:      Cost of moving from one state to another without eating a candy or being eaten by a ghost

        eat_cost:       Cost of eating a candy

        power:          Power parameter for the ghost moves, the higher the power the more the ghosts will try to get closer to pacman, power = 0 means the ghosts move randomly
        
        control_accuracy: Flag to enable the control of the accuracy of the value function from the optimum through the epsilon parameter

        logging:        Flag to enable logging of the algorithm steps
        """
        # Save the parameters from the enumerator
        self.map = enumerator.map
        self.states = enumerator.states
        self.initial_state = enumerator.initial_state
        self.number_of_movables = enumerator.number_of_movables
        self.number_of_ghosts = enumerator.number_of_ghosts
        self.number_of_candies = enumerator.number_of_candies
        self.possible_positions = enumerator.possible_positions
        self.candies_positions = enumerator.candies_positions
        self.possible_states = enumerator.number_of_possible_states
        self.filename = enumerator.filename
        self.power = power
        self.control_accuracy = control_accuracy

        # save value iteration hyperparameters

        # discount factor
        self.alpha = alpha
            # If we want to control the accuracy of the value function from the optimum through the epsilon parameter
        if self.control_accuracy:
            self.epsilon = epsilon
            self.delta = ((1 - self.alpha)/self.alpha) * self.epsilon
        else:
            # convergence threshold
            self.delta = delta
            self.epsilon = (self.alpha/(1-self.alpha)) * self.delta

        # cost function parameters
        self.lose_cost = lose_cost
        self.win_cost = win_cost
        self.move_cost = move_cost
        self.eat_cost = eat_cost
        
        # possible moves in a 2D grid
        self.moves = {
            0: (0, -1), # up
            1: (0, 1),  # down
            2: (-1, 0), # left
            3: (1, 0),  # right
            4: (0, 0)   # stay
        }

        # Dynamic programming variables
        self.policy = [0] * len(self.states)
        self.previous_value_function = {}
        self.value_function = {}
        
        # Initialize the value function
        self.previous_value_function = {tuple(s): 0 for s in self.states}
        self.value_function = {tuple(s): 0 for s in self.states}

        self.iterations = 1

        # Logging flag
        self.logging = logging


    def g(self, state, eaten):
        # if pacman is eaten by a ghost
        if state[0] in state[1:self.number_of_movables]:
            return self.lose_cost
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
            
            # If the ghost is in any of those neighboring cells, apply the lose cost
            if (ghost_x, ghost_y) in neighbors:
                return self.lose_cost
            
        # if pacman ate a candy
        if eaten:
            if sum(state[self.number_of_movables:]) == 0:
                return self.win_cost
            return self.eat_cost
        # otherwise
        return self.move_cost
    
    def store_policy(self, filename=""):
        if filename == "": filename = self.filename
        with open(f"./policies/{filename}_policy.txt", "w") as file:
            for s in range(len(self.states)):
                file.write(f"{self.states[s]}-->{self.moves[self.policy[s]]}\n")

    def store_value_function(self, filename=""):
        if filename == "": filename = self.filename
        with open(f"./value_functions/{filename}_value_function.txt", "w") as file:
            for key, value in self.value_function.items():
                file.write(f"{key} : {value}\n")

    def run(self):
        if self.logging: 
            print(f"Value iteration started with alpha={self.alpha}, epsilon={self.epsilon}, delta={self.delta}, control_accuracy={self.control_accuracy}")
            print(f"||V_(i+1) - V*||inf < {self.epsilon}  if  ||V_(i+1) - V_(i)||inf < {self.delta}\n")

        difference_norm = self.delta + 1
        while difference_norm >= self.delta:
            if self.logging: print(f"Iteration {self.iterations}", end=" - ")

            previous_value_function = self.value_function.copy()
            for s in range(len(self.states)):
                expected_values = [float("inf")] * len(self.moves)
                current_state = self.states[s]
                for pacman_action in self.moves:    
                    next_state, eaten = pacman_move(current_state, self.moves[pacman_action], self.number_of_movables, self.candies_positions, self.map)
                    # if this is not a valid action skip it
                    if next_state == False or (is_terminal(current_state, self.number_of_movables) and pacman_action != 4):
                        continue
                    # next_state is incomplete, only accounts for pacman move and eating a candy

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

                    # next_states contains all possible following states given the pacman action and the stochastic moves of the ghosts
                    stage_costs = [self.g(state, eaten) for state in next_states]

                    next_state_values = [previous_value_function[tuple(s)] for s in next_states]

                    # Calculate the expected value of the stage cost and the value of the next state
                    cost_expected_value = sum([stage_costs[i] * ghosts_action_permutations_pmfs[i] for i in range(len(next_states))])
                    value_expected_value = sum([next_state_values[i] * ghosts_action_permutations_pmfs[i] for i in range(len(next_states))])

                    expected_values[pacman_action] = (cost_expected_value + self.alpha * value_expected_value)
                
                self.policy[s] = argmin(expected_values)
                self.value_function[tuple(current_state)] = min(expected_values)
            
            difference_norm = diff_norm(self.value_function, previous_value_function)
            self.iterations += 1
            if self.logging: print(f"Norm of the difference: {difference_norm}")

        if self.logging: print(f"Value iteration converged after {self.iterations} iterations")


class Game:
    def __init__(self, value_iterator, pretrained=True, tile_size=32, fps=10, power=None, logging=False, measure_performance=False):
        """
        Game class to run the Pacman game with the policy learned from the value iteration algorithm (requires Value_iterator instance)

        value_iterator: Value_iterator instance, built as Value_iterator(enumerator, alpha, delta, epsilon, lose_cost, win_cost, move_cost, eat_cost, power, control_accuracy, logging)

        pretrained:     Flag to load the policy from a txt file instead of the value iterator instance

        tile_size:      Size of the tiles in the game window

        fps:            Frames per second of the game

        power:          Power parameter for the ghost moves, the higher the power the more the ghosts will try to get closer to pacman, 
                        power = 0 means the ghosts move randomly with uniform probability wrt the possible moves
                        power = 1 means the ghosts move with a probability proportional to the inverse of the manhattan distance to pacman
                        power = 2 means the ghosts move with a probability proportional to the inverse of the square of the manhattan distance to pacman
                        and so on...

        logging:        Flag to enable logging of the game steps
        """
        # Save the parameters from the value iterator
        self.map = value_iterator.map
        self.possible_positions = value_iterator.possible_positions
        self.states = value_iterator.states
        self.initial_state = value_iterator.initial_state
        self.current_state = self.initial_state.copy()
        self.number_of_movables = value_iterator.number_of_movables
        self.candies_positions = value_iterator.candies_positions
        self.moves = value_iterator.moves
        self.moves_to_policy = {v: k for k, v in self.moves.items()}    

        self.logging = logging
        self.measure_performance = measure_performance

        # Save the map filename
        self.map_name= value_iterator.filename

        # Initialize the power parameter
        if power == None:
            self.power = value_iterator.power
        else:
            self.power = power
        
        # Load the policy from a txt file or from the value iterator instance
        if pretrained:
            self.filename = value_iterator.filename
            self.policy = [0] * len(self.states)
            self.load_policy()
            if self.logging: print(f"Policy loaded from {self.filename}_policy.txt")
        else:
            self.policy = value_iterator.policy
            if self.logging: print("Policy loaded from the value iterator instance")

            # Check if the policy is loaded from a pretrained value iterator (Value_iterator policy is initialized as a list of zeros)
            sum_check = sum([p for p in self.policy])
            if sum_check == 0:
                if self.logging: 
                    print("The policy is not loaded from a pretrained value iterator")
                    print("Please run the value iterator algorithm to generate a policy")
                raise ValueError("The policy is not loaded from a pretrained value iterator")


        # Save the game parameters
        self.tile_size = tile_size
        self.fps = fps
        
        if not self.measure_performance:
            # Initialize the screen
            self.screen = self.init_screen()

            # Game images initialization
            self.pacman_image = None
            self.ghost_image = None
            self.candy_image = None
            self.floor_image = None
            self.wall_image = None
            self.init_images()

        # At test time, account for a minimum threshold in seconds, the number of candies eaten and the number of moves
        if self.measure_performance:
            self.loop_till_loss = True
            self.logging = False
            self.min_threshold = 600 # moves
            self.max_threshold = 6000 # moves
            self.candies_eaten = 0
            self.number_of_moves = 0
            self.efficeincy_ratio = 0

            self.alpha = value_iterator.alpha
            self.epsilon = value_iterator.epsilon
            self.delta = value_iterator.delta
            self.lose_cost = value_iterator.lose_cost
            self.win_cost = value_iterator.win_cost
            self.move_cost = value_iterator.move_cost
            self.eat_cost = value_iterator.eat_cost
            self.training_power = value_iterator.power


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

    def load_policy(self, filename=""):
        if filename == "": filename = self.filename
        with open(f"./policies/{filename}_policy.txt", "r") as file:
            for i in range(len(self.states)):
                line = file.readline()
                state, action = line.split("-->")
                action = action[1:-2].split(", ")
                action = tuple([int(action[0]), int(action[1])])
                self.policy[i] = self.moves_to_policy[action]

    def draw_map(self):
        # Draw walls and floors
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                if self.map[y][x] == 1:
                    self.screen.blit(self.wall_image, (x * self.tile_size, y * self.tile_size))
                else: 
                    self.screen.blit(self.floor_image, (x * self.tile_size, y * self.tile_size))

        # Draw candies            
        for candy_index in self.candies_positions:
            if self.current_state[candy_index] == 1:
                self.screen.blit(self.candy_image, (self.candies_positions[candy_index][0] * self.tile_size, self.candies_positions[candy_index][1] * self.tile_size))

        # Draw pacman 
        self.screen.blit(self.pacman_image, (self.current_state[0][0] * self.tile_size, self.current_state[0][1] * self.tile_size))

        # Draw ghosts
        if self.number_of_movables > 1:
            self.screen.blit(self.first_ghost_image, (self.current_state[1][0] * self.tile_size, self.current_state[1][1] * self.tile_size))
        for ghost_index in range(2, self.number_of_movables):
            self.screen.blit(self.ghost_image, (self.current_state[ghost_index][0] * self.tile_size, self.current_state[ghost_index][1] * self.tile_size))

    def run(self, ghost_controlled=False, loop_till_loss=False, measure_filename=""):
        clock = pygame.time.Clock()
        running = True

        # reset measures for multiple tests
        if self.measure_performance:
            self.candies_eaten = 0
            self.number_of_moves = 0
            self.efficeincy_ratio = 0

        if not self.measure_performance:
            # Before starting the game, display the logo for 2 seconds
            logo = pygame.image.load("./images/logo.png")
            logo = pygame.transform.scale(logo, (len(self.map[0]) * self.tile_size, len(self.map) * self.tile_size))
            self.screen.blit(logo, (0, 0))
            pygame.display.flip()
            sleep(3)

        # Randomize the initial positions of all ghosts
        for ghost_index in range(1, self.number_of_movables):
            new_ghost_position = self.possible_positions[random.choice(len(self.possible_positions))]
            # Avoid placing a ghost on pacman's position
            while new_ghost_position == self.current_state[0]:
                new_ghost_position = self.possible_positions[random.choice(len(self.possible_positions))]
            self.current_state[ghost_index] = new_ghost_position

        # Dictionary of pacman images based on action index
        pacman_images = {
            0: pygame.image.load("./images/u_pacman.png"),
            1: pygame.image.load("./images/d_pacman.png"),
            2: pygame.image.load("./images/l_pacman.png"),
            3: pygame.image.load("./images/r_pacman.png"),
            4: pygame.image.load("./images/r_pacman.png"),
        }

        while running:
            if self.measure_performance:
                if self.number_of_moves % 100 == 0:
                    if measure_filename == "": print(f"Simulated number of moves: {self.number_of_moves}")
                if self.number_of_moves == self.max_threshold:
                    self.efficeincy_ratio = self.candies_eaten / self.number_of_moves
                    if measure_filename == "": print(f"Test passed - Maximum number of moves ({self.max_threshold}) reached\n\tPacman efficiency ratio: {self.efficeincy_ratio}\n\tCandies eaten: {self.candies_eaten}")
                    running = False
                    clock.tick(self.fps)
                    continue

            key_pressed = not ghost_controlled # track whether a KEYDOWN happened

            # Draw current state every frame, so the window remains responsive 
            if not self.measure_performance:
                self.draw_map()
                pygame.display.flip()

                # Collect events 
                events = pygame.event.get()

                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        # If any key was pressed, we consider it a turn
                        key_pressed = True

            # If no key was pressed, we do nothing and skip to next iteration 
            if not key_pressed:
                # Limit CPU usage even when not moving
                clock.tick(self.fps)
                continue

            # If we get here, it means a key was pressed, so we process a turn
            # Check win/lose conditions 
            if loop_till_loss:
                # In the particular case of a single candy, respawn it and move pacman to a random position not on a ghost
                if len(self.candies_positions) == 1 and is_win_terminal(self.current_state, self.number_of_movables):
                    self.current_state[self.number_of_movables] = 1
                    new_pacman_position = self.possible_positions[random.choice(len(self.possible_positions))]
                    while new_pacman_position in self.current_state[1:] or new_pacman_position == self.candies_positions[self.number_of_movables]:
                        new_pacman_position = self.possible_positions[random.choice(len(self.possible_positions))]
                    self.current_state[0] = new_pacman_position
                else:
                    # Respawn one or more candies
                    while is_win_terminal(self.current_state, self.number_of_movables):
                        for candy_index in self.candies_positions:
                            if random.choice(2) == 1 and self.current_state[0] != self.candies_positions[candy_index]:
                                self.current_state[candy_index] = 1

            elif is_win_terminal(self.current_state, self.number_of_movables):
                print("Game over - You won")
                running = False
                sleep(2)
                clock.tick(self.fps)

                if self.logging: 
                    print(f"Wins, terminal state: {self.current_state}")

                continue

            if is_lose_terminal(self.current_state, self.number_of_movables):
                # Pacman lost before the maximum threshold
                if self.measure_performance:
                    self.efficeincy_ratio = self.candies_eaten / self.number_of_moves
                    if measure_filename == "": print(f"Test failed - Pacman eaten by a ghost after {self.number_of_moves} moves\n\tPacman efficiency ratio: {self.efficeincy_ratio}\n\tCandies eaten: {self.candies_eaten}")
                    running = False
                    clock.tick(self.fps)
                    continue

                print("Game over - You lost")
                running = False
                sleep(2)
                clock.tick(self.fps)

                if self.logging:
                    print(f"Defeat, terminal state: {self.current_state}")

                continue

            # Move pacman according to the policy
            state_index = self.states.index(self.current_state)
            action = self.policy[state_index]

            next_state, eaten = pacman_move(self.current_state, self.moves[action], self.number_of_movables, self.candies_positions, self.map)

            # Account for the number of moves
            if self.measure_performance:
                self.number_of_moves += 1
                self.candies_eaten += int(eaten)

            if self.logging: 
                print(f"Pacman action: {action} triggered transition from State: {self.current_state} to State: {next_state}")

            # Update Pac-Man image
            self.pacman_image = pygame.transform.scale(pacman_images[action], (self.tile_size, self.tile_size))

            # If the action was invalid, skip (ignore the pressed key)
            if not next_state:
                print("Invalid action")
                sleep(1)
                clock.tick(self.fps)

                if self.logging:
                    print(f"Invalid action: {action}")

                continue

            # Ghost movement
            if ghost_controlled:
                # Player controls the first ghost
                ghost_new_pos = next_state[1]  # current ghost 1 position

                # Re-check the same events array (it has the key that was pressed):
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if (event.key == pygame.K_UP and self.map[ghost_new_pos[1] - 1][ghost_new_pos[0]] != 1):
                            ghost_new_pos = (ghost_new_pos[0], ghost_new_pos[1] - 1)
                        elif (event.key == pygame.K_DOWN and self.map[ghost_new_pos[1] + 1][ghost_new_pos[0]] != 1):
                            ghost_new_pos = (ghost_new_pos[0], ghost_new_pos[1] + 1)
                        elif (event.key == pygame.K_LEFT and self.map[ghost_new_pos[1]][ghost_new_pos[0] - 1] != 1):
                            ghost_new_pos = (ghost_new_pos[0] - 1, ghost_new_pos[1])
                        elif (event.key == pygame.K_RIGHT and self.map[ghost_new_pos[1]][ghost_new_pos[0] + 1] != 1):
                            ghost_new_pos = (ghost_new_pos[0] + 1, ghost_new_pos[1])

                next_state[1] = ghost_new_pos

                if self.logging:
                    print(f"Controlled ghost 1 action triggered transition to State: {next_state}")

                # Stochastic moves for additional ghosts, if any
                if self.number_of_movables > 2:
                    possible_ghosts_actions = []
                    ghosts_actions_pmfs = []
                    for ghost_index in range(2, self.number_of_movables):
                        action_list, pmf = ghost_move_manhattan(next_state, ghost_index, self.moves, self.map, self.power)
                        possible_ghosts_actions.append(action_list)
                        ghosts_actions_pmfs.append(pmf)

                    # Build permutations
                    permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
                    permuted_pmfs = [list(p) for p in product(*ghosts_actions_pmfs)]
                    ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]

                    # Sample a combination
                    ghosts_actions = permuted_actions[random.choice(len(permuted_actions), p=ghosts_action_permutations_pmfs)]

                    # Apply the chosen moves
                    updated_positions = []
                    for i, ga in enumerate(ghosts_actions):
                        # ghost_index = 2 + i
                        gx, gy = next_state[2 + i]
                        updated_positions.append((gx + ga[0], gy + ga[1]))

                    # Rebuild next_state (Pac-Man = next_state[0], ghost1= [1])
                    next_state = [next_state[0], next_state[1]] + updated_positions + next_state[self.number_of_movables:]

                    if self.logging:
                        print(f"Ghosts actions triggered transition to State: {next_state}")

            else:
                # If ghost_controlled = False, all ghosts move stochastically
                possible_ghosts_actions = []
                ghosts_actions_pmfs = []
                for ghost_index in range(1, self.number_of_movables):
                    action_list, pmf = ghost_move_manhattan( next_state, ghost_index, self.moves, self.map, self.power)
                    possible_ghosts_actions.append(action_list)
                    ghosts_actions_pmfs.append(pmf)
                
                # Build permutations
                permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
                permuted_pmfs = [p for p in product(*ghosts_actions_pmfs)]
                ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]

                # Randomly choose the configuration
                ghosts_actions = permuted_actions[random.choice(len(permuted_actions), p=ghosts_action_permutations_pmfs) ]

                updated_positions = []
                for i, ga in enumerate(ghosts_actions):
                    gx, gy = next_state[1 + i]
                    updated_positions.append((gx + ga[0], gy + ga[1]))

                next_state = [next_state[0]] + updated_positions + next_state[self.number_of_movables:]

                if self.logging:
                    print(f"Ghosts actions triggered transition to State: {next_state}")

            # Update current state
            self.current_state = next_state

            # Limit the frame rate
            clock.tick(self.fps)

        if self.measure_performance:
            params = f"efficiency = {self.efficeincy_ratio}, number_of_moves = {self.number_of_moves}, candies_eaten = {self.candies_eaten} - alpha = {self.alpha}, delta = {self.delta}, epsilon = {self.epsilon}, lose_cost = {self.lose_cost}, win_cost = {self.win_cost}, move_cost = {self.move_cost}, eat_cost = {self.eat_cost}, training_power = {self.training_power}, game_power = {self.power}"
            if self.number_of_moves < self.min_threshold:
                with open("./parallel_jobs/"+measure_filename+"_under_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")
            elif self.number_of_moves < self.max_threshold:
                with open("./parallel_jobs/"+measure_filename+"_between_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")
            else:
                with open("./parallel_jobs/"+measure_filename+"_over_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")
                

        # Quit the game once we exit the loop 
        pygame.quit()
