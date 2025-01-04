# A Pac-man model for Dynamic Programming solution
from PIL import Image
import pprint
import numpy as np
from itertools import product
from copy import copy
from math import comb as binomial
import sys 

# Game Map attempt - read map from small png
img = Image.open('./maps/ez_map.png')

map = []
color_codes = {
    (255, 255, 255): 0, # white - floor
    (0, 0, 0): 1, # black - wall
    (255, 0, 0): 2, # red - candy
    (0, 255, 0): 3, # green - pacman
    (0, 0, 255): 4, # blue - ghost
}

w, h = img.size
for y in range(h):
    row = []
    for x in range(w):
        rgb = (img.getpixel((x, y))[0], img.getpixel((x, y))[1], img.getpixel((x, y))[2])
        row.append(color_codes[rgb])
    map.append(row)

print(f"Map height: {len(map)}")
print(f"Map width {len(map[0])}", end="\n\n")
pprint.pprint(map)
print()

# Acquire the current state as:
# a vector
# first tuple is the position of the pacman
# a series of tuple as the position of the ghosts
# lexicographic reading of the map with 1 if there is a candy, 0 if there was a candy

initial_state = [(0,0)] # contains a placeholder for the pacman
for y in range(h):
    for x in range(w):
        # Pacman
        if map[y][x] == 3:
            initial_state[0] = (x, y)
        # Ghosts
        elif map[y][x] == 4:
            initial_state.append((x, y))

# The state is now [pacman, ghost1, ghost2, ...]
number_of_movables = len(initial_state)

# Add the candies
candies_positions = {}
for y in range(h):
    for x in range(w):
        # Candy
        if map[y][x] == 2:
            initial_state.append(1)
            candies_positions[len(initial_state) - 1] = (x,y) # index of the candy in the state -> position in the map

number_of_candies = len(initial_state) - number_of_movables



# The state is now [pacman, ghost1, ghost2, ..., candy1, candy2, ...]
print(f"Initial state: {initial_state} of size {sys.getsizeof(initial_state)} bytes")
print(f"tuple size {sys.getsizeof(initial_state[0])} bytes")

# Calculate the possible x, y values for movables
possible_positions = []
for y in range(h):
    for x in range(w):
        if map[y][x] != 1:  # if it is not a wall
            possible_positions.append((x, y))

free_positions = len(possible_positions)    
number_of_ghosts = number_of_movables - 1

print(f"Number of movable positions in the map: {free_positions}")


#counting the number of states
number_of_states = 0

#accounting for all possible states with at least one candy not eaten
for i in range(1, number_of_candies + 1):
    number_of_states += (free_positions - i) * (free_positions ** number_of_ghosts) * binomial(number_of_candies, i)

#accounting for all possible states with all candies eaten (all candies are 0)
number_of_states += number_of_candies * (free_positions ** number_of_ghosts)

print(f"Number of possible states: {number_of_states}")

# couting the number of terminal states

#accounting for all possible winning terminal states (all candies are 0, pacman is not eaten by a ghost and pacman is on a candy)  
number_of_terminal_states = number_of_candies * ( (free_positions - 1) ** number_of_ghosts)

#accounting for all possible losing terminal states (pacman is eaten by a ghost)
# at least one ghost is on the same position of pacman and at least one candy is 1
for i in range(1, number_of_candies + 1):
    number_of_terminal_states += (free_positions - i) * ((2 ** number_of_ghosts) -1) * binomial(number_of_candies, i)

#accounting for all candies eaten (terminal state) and pacman is eaten by a ghost
number_of_terminal_states += number_of_candies * ((2 ** number_of_ghosts) - 1)


# there are states in which pacman stands on a candy and the candy is 1 nonetheless
# this is a state that is not possible in the game and must be subtracted
# the number of such states is equal to the number of candies for each possible position of the ghosts


def is_terminal(state):
    return sum(state[number_of_movables:]) == 0 or state[0] in state[1:number_of_movables]

# For Dynamic Programming we need to enumerate each state
states = []
terminal_states_count = 0
# add all pacman moves with ghosts in the initial position
for pacman_position in possible_positions:
    new_state = [pacman_position] + initial_state[1:]
    # -- Candy eaten check
    for candy_index in range(number_of_movables, len(initial_state)):
        if pacman_position == candies_positions[candy_index]:
            new_state[candy_index] = 0
    if is_terminal(new_state):
        terminal_states_count += 1
    states.append(new_state)


print(f"Number of states with everything fixed except Pac-man:\n-\tGeneric states: {len(states)}\n-\tOf which terminal: {terminal_states_count}")

# add all ghosts moves for all pacman moves
for ghost_index in range(1, number_of_movables):
    for s in range(len(states)):
        #pacman_position = states[s][0]
        for ghost_position in possible_positions:
            if ghost_position == initial_state[ghost_index]:
                continue    # Such state already exists
            new_state = states[s][:ghost_index] + [ghost_position] + states[s][ghost_index + 1:]
            if is_terminal(new_state):
                terminal_states_count += 1
            states.append(new_state)

print(f"Number of states with moving ghosts and Pac-man:\n-\tGeneric states: {len(states)}\n-\tOf which terminal: {terminal_states_count}")
# ! Note: there is still space for state reduction... Having ghosts occupying opposite positions is functionally equal !
# * I think this should not stand a big problem but further optimization in larger maps might require this *


# add all candies states
for candy_index in range(number_of_movables, len(initial_state)):
    for s in range(len(states)):
        if states[s][candy_index] == 1:
            new_state = states[s][0:candy_index] + [0] + states[s][candy_index + 1:]
            if sum(new_state[number_of_movables:]) == 0:
                if states[s][0] in [candies_positions[i] for i in range(number_of_movables, len(initial_state))]:
                    terminal_states_count += 1
                    states.append(new_state)
            else:
                # pacman is eaten by a ghost and a candy is changed to 0 --> this is a terminal state
                if is_terminal(new_state):
                    terminal_states_count += 1
                states.append(new_state)
 

print(f"Number of states accounting for candies:\n-\tGeneric states: {len(states)}\n-\tOf which terminal: {terminal_states_count}\n\t")

import sys
print(f"{sys.getsizeof(states)} bytes")


# A dictionary for possible moves
moves = {
    0: (0, -1), # up
    1: (0, 1),  # down
    2: (-1, 0), # left
    3: (1, 0),  # right
    4: (0, 0)   # stay
}

# A function to valdiate pacman actions and return a state

def pacman_move(state, action):
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
            if new_position == candies_positions[candy_index]:
                new_state[candy_index] = 0
                eaten=True

        return new_state, eaten

# A function to validate ghost actions and return a list of possible actions
def ghost_move(state, ghost_index):
    ghost_position = state[ghost_index]
    actions = [moves[a] for a in moves]
    possible_actions = []
    # exclude stay action for ghosts
    for a in actions[:-1]:
        new_position = (ghost_position[0] + a[0], ghost_position[1] + a[1])
        if map[new_position[1]][new_position[0]] == 1:  # if it is a wall
            continue
        possible_actions.append(a)
    return possible_actions


# cost function
def g(state, eaten):
    if state[0] in state[1:number_of_movables]:
        return 1000
    # se il pacman ha mangiato una caramella il costo è -10
    if eaten:
        if sum(state[number_of_movables:]) == 0:
            return -500
        return -10
    # altrimenti il costo è
    return 1

def diff_norm(dict_1, dict_2):
    if len(dict_1) != len(dict_2):
        print("Norm of 2 differently sized dictionaries attempted!")
        return None
    
    return max([dict_1[s] - dict_2[s] for s in dict_1])


def is_win_terminal(state):
    return sum(state[number_of_movables:]) == 0 and state[0] not in state[1:number_of_movables]

def is_lose_terminal(state):
    return state[0] in state[1:number_of_movables]
'''
win_terminal_count = 0
lose_terminal_count = 0
for state in states:
    if is_win_terminal(state):
        win_terminal_count += 1
    elif is_lose_terminal(state):
        lose_terminal_count += 1

print(f"\n\nWin terminal states: {win_terminal_count} - Lose terminal states: {lose_terminal_count}\n\n")
print(f"Total terminal states: {win_terminal_count + lose_terminal_count}")


# Value function
previous_value_function = {tuple(s) : 1 for s in states}
value_function = {tuple(s) : 0 for s in states}

# Policy
policy = np.zeros(shape=(len(states)), dtype=int)

# Training
# say delta is the minimum change from one generation to the next
delta = 0.01

# discount factor
alpha = 0.9


# training loop
gen = 1
difference_norm = 1
while difference_norm >= delta:
    print(f"\nTraining generation {gen}:")
    previous_value_function = copy(value_function)
    for s in range(len(states)):
        # Update policy
        # Find the arg min, w.r.t. the pacman action, of the expected value of stage cost + value of state
        # Fix a valid pacman move
        expected_values = [float("inf")] * len(moves)
        for pacman_action in moves:
            next_state, eaten = pacman_move(states[s], moves[pacman_action])
            # if this is not a valid action skip it
            if next_state == False or (is_terminal(states[s]) and pacman_action != 4):
                expected_values[pacman_action] = float("inf")
                continue
            # next_state is incomplete, only accounts for pacman move and eating a candy

            possible_ghosts_actions = [ghost_move(next_state, ghost_index) for ghost_index in range(1,number_of_movables)]
            
            permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]

            next_states = []
            for actions in permuted_actions:
                ghosts_state = []
                for i in range(len(actions)):
                    ghosts_state.append((actions[i][0] + next_state[1+i][0], actions[i][1] + next_state[1+i][1]))
                next_states.append([next_state[0]] + ghosts_state + next_state[number_of_movables:])
            
            # next_states contains all possible following states given the pacman action and the stochastic moves of the ghosts
            
            stage_costs = [g(state, eaten) for state in next_states]
            next_state_values = [previous_value_function[tuple(s)] for s in next_states]

            # with uniform probability every ghost configuration is equally probable --> expected value is a mean
            cost_expected_value = sum(stage_costs)/len(next_states)
            value_expected_value = sum(next_state_values)/len(next_states)

            expected_values[pacman_action] = (cost_expected_value + alpha * value_expected_value)

        policy[s] = np.argmin(expected_values)
        value_function[tuple(states[s])] = min(expected_values)

    gen += 1
    difference_norm = diff_norm(previous_value_function, value_function)
    #pprint.pprint(value_function)
    print(f"Gen {gen-1} to Gen {gen} improvement: {difference_norm}")


'''
"""
for i in range(len(states)):
    print(f"{states[i]} --> {moves[policy[i]]}")
"""


def store(value_function, policy, map_name):
    with open(map_name + '_value_function.txt', 'w') as f:
        for key, value in value_function.items():
            f.write(f"{key} : {value}\n")
    with open(map_name + '_policy.txt', 'w') as f:
        for i in range(len(states)):
            f.write(f"{states[i]} --> {moves[policy[i]]}\n")


# dictionary to convert actions to policy 
moves_to_policy = {v: k for k, v in moves.items()}


def load(map_name):
    #counting the number of states == number of lines in the file and loading the value function
    value_function = {}
    states_number = 0
    with open('value_functions/' +map_name + '_value_function.txt', 'r') as f:
        for line in f:
            key, value = line.split(" : ")
            value_function[key] = float(value)
            states_number += 1
    #loading the policy
    policy = np.zeros(shape=(states_number), dtype=int)
    with open('policies/'+map_name + '_policy.txt', 'r') as f:
        for i in range(states_number):
            line = f.readline()
            _ , action = line.split("-->")
            #casting the action to a tuple
            action = action[1:-2].split(", ")
            action = tuple([int(action[0]), int(action[1])])
            policy[i] = moves_to_policy[action]
    return value_function, policy



# play function using is_terminal function
def play():
    state = initial_state
    print(f"Initial state: {state}")
    num_moves = 0
    while is_terminal(state) == False:
        print(f"Current state: {state}")
        pacman_action = policy[states.index(state)]
        next_state, eaten = pacman_move(state, moves[pacman_action])
        print(f"Pacman moves {moves[pacman_action]}")
        if next_state == False:
            print("Invalid move, retry")
            continue
        print(f"State after pacman move: {next_state}")

        possible_ghosts_actions = [ghost_move(next_state, ghost_index) for ghost_index in range(1,number_of_movables)]
        #saple ghost actions
        permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
        #choose random ghost action
        ghosts_actions = permuted_actions[np.random.choice(len(permuted_actions))]
        #update next state with ghost actions
        next_state = [next_state[0]] + [(ghosts_actions[i][0] + next_state[1+i][0], ghosts_actions[i][1] + next_state[1+i][1]) for i in range(len(ghosts_actions))] + next_state[number_of_movables:]        

        print(f"State after ghost move: {next_state}")

        state = next_state
        
        if eaten:
            print("Pacman ate a candy!")
        num_moves += 1
    print(f"Game ended in {num_moves} moves")


#store(value_function, policy, 'pac-man-6c-1g')

value_function, policy = load('ez_map')
print(policy)
#play()



# graphical visualization of the game
import pygame
import time

# Initialize pygame
pygame.init()

# Constants
TILE_SIZE = 32  # Size of each tile
FPS = 10  # Frames per second

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Load the map
map_width = len(map[0])
map_height = len(map)

# Initialize the screen
screen = pygame.display.set_mode((map_width * TILE_SIZE, map_height * TILE_SIZE))
pygame.display.set_caption("Pac-Man Visualization")

# Load assets (optional)
pacman_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
pacman_image.fill(GREEN)
ghost_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
ghost_image.fill(BLUE)
wall_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
wall_image.fill(BLACK)
candy_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
candy_image.fill(RED)
floor_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
floor_image.fill(WHITE)

# Render the game map
def draw_map(game_map, state):
    for y in range(len(game_map)):
        for x in range(len(game_map[0])):
            tile = game_map[y][x]
            if tile == 1:  # Wall
                screen.blit(wall_image, (x * TILE_SIZE, y * TILE_SIZE))
            else:  # Floor
                screen.blit(floor_image, (x * TILE_SIZE, y * TILE_SIZE))
    
    # Draw candies
    for candy_index in candies_positions:
        if state[number_of_movables + candy_index - number_of_movables] == 1:  # Candy present
            x, y = candies_positions[candy_index]
            screen.blit(candy_image, (x * TILE_SIZE, y * TILE_SIZE))
    
    # Draw Pac-Man
    pacman_x, pacman_y = state[0]
    screen.blit(pacman_image, (pacman_x * TILE_SIZE, pacman_y * TILE_SIZE))
    
    # Draw ghosts
    for ghost_index in range(1, number_of_movables):
        ghost_x, ghost_y = state[ghost_index]
        screen.blit(ghost_image, (ghost_x * TILE_SIZE, ghost_y * TILE_SIZE))

# Visualize the game
def visualize_game(initial_state, policy, fps=FPS):
    clock = pygame.time.Clock()
    running = True
    state = initial_state.copy()
    # randomize the ghosts positions
    for i in range(1, number_of_movables):
        new_position = possible_positions[np.random.choice(len(possible_positions))]
        while new_position == state[0]:
            new_position = possible_positions[np.random.choice(len(possible_positions))]
        state[i] = possible_positions[np.random.choice(len(possible_positions))]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Render the map and agents
        draw_map(map, state)
        pygame.display.flip()

        # Check if the game is over
        if is_win_terminal(state):
            # respawn a random number of candies
            for i in range(number_of_movables, len(initial_state)):
                if np.random.choice([True, False]) and state[0] != candies_positions[i]:
                    state[i] = 1

            #print("You won!")
            #time.sleep(2)
            #running = False
        elif is_lose_terminal(state):
            print("You lost!")
            time.sleep(2)
            running = False

        # Move Pac-Man based on policy
        state_index = states.index(state)   # o(n)
        action = policy[state_index]
        next_state, _ = pacman_move(state, moves[action])

        print(f"action: {moves[action]}")

        if next_state==False:
            print("Invalid move")
            time.sleep(2)
            continue

        possible_ghosts_actions = [ghost_move(next_state, ghost_index) for ghost_index in range(1,number_of_movables)]
        #saple ghost actions
        permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
        #choose random ghost action
        ghosts_actions = permuted_actions[np.random.choice(len(permuted_actions))]
        #update next state with ghost actions
        next_state = [next_state[0]] + [(ghosts_actions[i][0] + next_state[1+i][0], ghosts_actions[i][1] + next_state[1+i][1]) for i in range(len(ghosts_actions))] + next_state[number_of_movables:]        

        state = next_state

        # Cap the frame rate
        clock.tick(fps)

    pygame.quit()

# Run the visualization
visualize_game(initial_state, policy, fps=10)
