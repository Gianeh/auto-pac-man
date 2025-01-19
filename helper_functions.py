# Some helper methods to check states, stage costs, norm of the difference
import heapq
from PIL import Image
from math import comb as binomial
from pprint import pprint
import pygame
from time import sleep

# Check if a state is terminal, with no regard to impossibility of the state
def is_terminal(state, number_of_movables):
    return sum(state[number_of_movables:]) == 0 or state[0] in state[1:number_of_movables]

# check if a state is a win terminal state
def is_win_terminal(state, number_of_movables):
    return sum(state[number_of_movables:]) == 0 and state[0] not in state[1:number_of_movables]

# check if a state is a lose terminal state
def is_lose_terminal(state, number_of_movables):
    return state[0] in state[1:number_of_movables]

# Evaluate if a pacman move is valid and return a new state
def pacman_move(state, action, number_of_movables, candies_positions, map):
    eaten=False
    # current position
    pacman_position = state[0]
    # next position
    new_position = (pacman_position[0] + action[0], pacman_position[1] + action[1])
    # if invalid return False
    if map[new_position[1]][new_position[0]] == 1:
        return False, eaten
    else:
        new_state = [new_position] + state[1:]
        # candy eaten check
        for candy_index in range(number_of_movables, len(state)):
            if new_position == candies_positions[candy_index] and state[candy_index] == 1:
                new_state[candy_index] = 0
                eaten=True

        return new_state, eaten

# Create the Probability Mass Function for the selected ghost moves according to the power 
def build_pmf(distances, power):
    # Calculate the inverse of each number ensuring that the distance is never 0 (to avoid division by 0 a +1 is added to the denominator) 
    inverse_distances = [1 / (distance**power + 1) for distance in distances]
    
    # Normalize the inverses to sum to 1
    total = sum(inverse_distances)
    pmf = [inv_dist / total for inv_dist in inverse_distances]
    
    return pmf

# Compute the A* path distance from current ghost position to pacman position
def a_star_distance(game_map, start, goal):
    if start == goal:
        return 0

    # Manhattan distance
    def heuristic(pos1, pos2):
        (x1, y1), (x2, y2) = pos1, pos2
        return abs(x1 - x2) + abs(y1 - y2)

    visited = set()
    # Priority queue stores (f, g, (x, y)), where f = g + h
    priority_queue = []
    start_h = heuristic(start, goal)
    heapq.heappush(priority_queue, (start_h, 0, start))  # f, g, position

    while priority_queue:
        # this returns the tuple with the smallest f
        f, g, current = heapq.heappop(priority_queue)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return g  # g is cost-so-far, i.e. actual distance

        current_x, current_y = current
        for new_x, new_y in [(current_x+1, current_y), (current_x-1, current_y), (current_x, current_y+1), (current_x, current_y-1)]:
            # Check boundaries and walls
            if 0 < new_y < len(game_map)-1 and 0 < new_x < len(game_map[0])-1:
                if game_map[new_y][new_x] != 1 and (new_x, new_y) not in visited:
                    new_g = g + 1
                    new_h = heuristic((new_x, new_y), goal)
                    new_f = new_g + new_h
                    heapq.heappush(priority_queue, (new_f, new_g, (new_x, new_y)))

    # Goal not reachable
    return 9999

# Compute the possible actions for a ghost and the Probability Mass Function for each action according to the power
def ghost_move_pathfinding(state, ghost_index, moves, game_map, power=1):
    ghost_position = state[ghost_index]

    # Collect valid actions (exclude stay action for ghosts, if you wish)
    possible_actions = []
    a_star_distances = []
    # Exclude the last item if it's 'stay'
    for action_key in list(moves.keys())[:-1]:
        dx, dy = moves[action_key]
        new_position = (ghost_position[0] + dx, ghost_position[1] + dy)

        # Skip if this new cell is a wall
        if game_map[new_position[1]][new_position[0]] == 1:
            continue

        possible_actions.append((dx, dy))

        # BFS distance from new_position to pac-man
        dist = a_star_distance(game_map, new_position, state[0])
        a_star_distances.append(dist)

    # Build pmf just like with manhattan_distance
    # e.g., if you have: build_pmf(distances, power)
    pmf = build_pmf(a_star_distances, power)

    return possible_actions, pmf

# Initializer class to start policy iteration algorithm and print the number of possible states and terminal states
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

# Helper class to render the game  
class Renderer:
    def __init__(self, initializer, tile_size=50, fps=10):
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

    def display_logo(self):
        logo = pygame.image.load("./images/logo.png")
        logo = pygame.transform.scale(logo, (len(self.map[0]) * self.tile_size, len(self.map) * self.tile_size))
        self.screen.blit(logo, (0, 0))
        pygame.display.flip()
        sleep(3)

    def clock_tick(self, fps=None):
        if fps is not None:
            self.clock.tick(fps)
        else:
            self.clock.tick(self.fps)

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
