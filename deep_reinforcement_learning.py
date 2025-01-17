# Classes for State counting and reinforcement learning training
from PIL import Image
from math import comb as binomial
from math import prod, sqrt
from pprint import pprint
from itertools import product
from random import choices, choice, random, sample
from time import sleep
import pygame
import os
from helper_functions import is_terminal, pacman_move, is_win_terminal, is_lose_terminal, ghost_move_pathfinding
import copy

# classes for neural network class
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def debug_log(self, q_values, loss):
    # 1) Check for inf/NaN in Q-network parameters & gradients
    bad_params = []
    bad_grads = []
    for name, param in self.Q.named_parameters():
        if param is not None:
            if torch.isnan(param).any() or torch.isinf(param).any():
                bad_params.append(name)
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                bad_grads.append(name)

    # 2) Summarize Q-values from the sampled batch
    # q_values is the result of: q_values = self.Q(current_states_tensor) before .gather(...)
    q_min = q_values.min().item()
    q_max = q_values.max().item()
    q_mean = q_values.mean().item()
    q_std = q_values.std().item()

    # 3) Print out debug info
    print("\n=== DEBUG LOG ===")
    print(f"Loss: {loss.item():.4f}")
    print(f"Q-values stats [batch before gather]: min={q_min:.4f}, max={q_max:.4f}, mean={q_mean:.4f}, std={q_std:.4f}")
    if len(bad_params) > 0:
        print(f"[WARNING] Found NaN/inf in parameters: {bad_params}")
    if len(bad_grads) > 0:
        print(f"[WARNING] Found NaN/inf in gradients: {bad_grads}")

    # 4) (Optional) Peek at a small sample of stored transitions
    if len(self.replay_buffer) > 0:
        sample_transitions = self.replay_buffer[-3:]  # Last 3 transitions stored
        print("Last few transitions in replay buffer:")
        for i, (s, a, r, s_next, done) in enumerate(sample_transitions):
            print(f"  Transition #{len(self.replay_buffer) - 3 + i}")
            print(f"    state={s}, action={a}, reward={r}, done={done}")
    print("=================\n")


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        # Example MLP: input -> hidden -> hidden -> output
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Flatten a state removing tuples
def encode_state(state, number_of_movables, candies_positions, map):
    # Positions
    coords = []
    for i in range(number_of_movables):
        coords.append(state[i][0])  # x
        coords.append(state[i][1])  # y
    
    # Candies
    #candies = state[number_of_movables:]
    # Distances to candies only if they are not eaten, 0 otherwise
    candies = []
    
    for i in range(number_of_movables, len(state)):
        # If the candy is active, append its position
        if state[i] == 1:
            position = candies_positions[i]
            candies.append(position[0])
            candies.append(position[1])
        # If the candy is eaten, append -1 for both coordinates
        else:
            candies.append(-1)
            candies.append(-1)
        candies.append(state[i]) # 1 if the candy is active, 0 otherwise

    # Walls in neighborhood
    neighbors = []
    pacman = state[0]
    for move in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        x = pacman[0] + move[0]
        y = pacman[1] + move[1]
        neighbors.append(1 if map[y][x] == 1 else 0)

    
    # Final flatten
    encoded = neighbors + coords + candies
    # Casting to float32 to enable torch operations with weights
    return torch.tensor(encoded, dtype=torch.float32)

# Class for policy iteration using Deep Q-Learning

class Neural_Policy_iterator:
    def __init__(self, initializer, renderer=None, max_episodes=1000, pretrained=False, alpha=1e-1, gamma=9e-1, epsilon=1e0, min_epsilon=5e-2, lose_reward=-1e3, win_reward=5e3, move_reward=-1, eat_reward=1e1, power=10, increasing_power=False, random_spawn=False, logging=False):
        """
        Generalized Policy Iteration algorithm using Deep Q-Learning with a MLP (requires State_initializer instance)

        initializer:        State_initializer instance, built as State_initializer(map_filename, logging)

        renderer:           Renderer object to visualize the graphics of the game

        max_episodes:       Number of episodes to run the algorithm
        
        alpha:              Learning rate, weights the new information against the old information during the update of the Q function

        gamma:              Discount factor, weights the future rewards against the current rewards

        epsilon:            Exploration rate, the higher the value the more the agent will explore the environment

        min_epsilon:        Minimum value for epsilon, the exploration rate will decay towards this value

        lose_reward:        reward of losing the game (eaten by a ghost)

        win_reward:         reward of winning the game (eating all candies)

        move_reward:        reward of moving from one state to another without eating a candy or being eaten by a ghost

        eat_reward:         reward of eating a candy

        power:              Power parameter for the ghost moves, the higher the power the more the ghosts will try to get closer to pacman, power = 0 means the ghosts move randomly
        
        increasing_power:   Flag to increase the power parameter every 500 episodes when epsilon reaches min_epsilon
        
        random_spawn:       Flag to randomly spawn at least one candy, ghosts and pacman at the beginning of each episode
        
        logging:            Flag to enable logging of the algorithm steps
        """
        # Save the initializer parameters
        self.map = initializer.map
        self.initial_state = initializer.initial_state
        self.number_of_movables = initializer.number_of_movables
        self.number_of_ghosts = initializer.number_of_ghosts
        self.number_of_candies = initializer.number_of_candies
        self.possible_positions = initializer.possible_positions
        self.candies_positions = initializer.candies_positions
        self.filename = initializer.filename

        self.power = power
        self.increasing_power = increasing_power
        self.random_spawn = random_spawn

        # save policy iteration hyperparameters
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon

        # How many transitions to sample from the replay buffer for each training step of the Q-network
        self.batch_size = 64

        # Number of trainig steps (of the Q-network) before updating the target network
        self.update_target_steps = 1000  

        # Counter for the number of learning steps of the Q-network
        self.learn_step_counter = 0

        # Replay buffer for experience replay
        self.replay_buffer = []
        self.replay_capacity = 1000000

        # reward function parameters
        self.lose_reward = lose_reward
        self.invalid_move_reward = lose_reward / 4
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

        # Initialize the MLP-based Q function and the target network
        self.Q = DQNNetwork(state_dim=len(encode_state(self.initial_state, self.number_of_movables, self.candies_positions, self.map)), action_dim=5) # including stay action
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()

        # Load pretrained weights
        if pretrained:
            self.load_Q()

        self.logging = logging

        # Renderer object to visualize the game
        self.renderer = renderer

        # Episode counter
        self.episodes = 0

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
   
    # Store the Q function weights in a file
    def store_Q(self):
        os.makedirs("./weights", exist_ok=True)
        torch.save(self.Q.state_dict(), f"./weights/{self.filename}_Q.pt")
    
    # Load the Q function weights from a file
    def load_Q(self):
        if os.path.exists(f"./weights/{self.filename}_Q.pt"):
            self.Q.load_state_dict(torch.load(f"./weights/{self.filename}_Q.pt", weights_only=True))
            if self.logging: 
                print(f"Pretrained weights loaded from ./weights/{self.filename}_Q.pt")
        else:
            print(f"No pretrained weights found at ./weights/{self.filename}_Q.pt")

    def pi(self, state):
        # exploration
        if random() < self.epsilon:
            return choice(list(self.moves.keys()))
        # exploitation
        else:
            state_tensor = encode_state(state, self.number_of_movables, self.candies_positions, self.map)
            with torch.no_grad():
                q_vals = self.Q(state_tensor)
                # return the index of the max Q value
                """if self.episodes % 100 == 0 and self.logging:
                    print(f"In State {state} the Q values are {q_vals}")"""
                return q_vals.argmax().item()

    def store_transition(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)
            # Use dqueue instead of list for better performance ???

    def sample_and_learn(self):
        # Don’t update if not enough transitions yet
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a random batch from replay buffer
        transitions = sample(self.replay_buffer, self.batch_size)
        
        # Convert them into batches of tensors
        current_states, actions, rewards, next_states, dones = zip(*transitions)

        # Encode states
        # State Matrix: every row is a current state of the transition
        current_states_tensor = torch.stack([encode_state(state, self.number_of_movables, self.candies_positions, self.map) for state in current_states])
        # Next state Matrix: every row is a next state of the transition
        next_states_tensor = torch.stack([encode_state(next_state, self.number_of_movables, self.candies_positions, self.map) for next_state in next_states])
        # Actions Tensor: every row is the action taken in the transition
        actions_tensor = torch.tensor(actions, dtype=torch.int64).view(-1, 1) 
        # Rewards Tensor: every row is the reward of the transition
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        # Terminal Tensor: every row is the terminal flag of the next state of the transition
        terminals_tensor = torch.tensor(dones, dtype=torch.int8).view(-1, 1)

        # Get current Q-values from main network
        q_values = self.Q(current_states_tensor)
        # Gather the Q-value for the chosen action
        q_a = q_values.gather(1, actions_tensor)


        # Compute Q target via target network
        with torch.no_grad():
            q_next = self.Q_target(next_states_tensor)
            q_next_max = q_next.max(dim=1, keepdim=True)[0]
            q_target = rewards_tensor + (1 - terminals_tensor) * self.gamma * q_next_max

        loss = torch.nn.MSELoss()(q_a, q_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network occasionally
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def run(self):
        if self.logging: 
            print(f"Running Policy Iteration algorithm for {self.max_episodes} episodes...\n")
            print(f"Episode {self.episodes}")
        
        action = 0 # Action placeholder for renderer object
        current_state = self.initial_state.copy()
        next_state = self.initial_state.copy()

        is_paused = False  # Pause state flag

        original_epsilon = self.epsilon

        n_games = 0     # Number of games played
        n_wins = 0      # Number of games won

        # Main loop of the policy iteration
        while self.episodes <= self.max_episodes:

            # Check if a key was pressed to lower or increase the fps of the renderer
            if self.renderer is not None:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.renderer.fps += 25
                        elif event.key == pygame.K_DOWN:
                            self.renderer.fps -= 25
                            if self.renderer.fps <= 0:
                                self.renderer.fps = 1
                        elif event.key == pygame.K_p:  # Toggle pause
                            is_paused = not is_paused
            
            # If paused, skip game updates and slow the loop
            if is_paused:
                if self.renderer is not None:
                    pygame.time.Clock().tick(10)  # Slow loop to save CPU
                continue

            # Render the training
            if self.renderer is not None:
                self.renderer.render(current_state, action)
            
            action = self.pi(current_state)

            next_state, eaten = pacman_move(current_state, self.moves[action], self.number_of_movables, self.candies_positions, self.map)

            invalid = False
            if next_state == False:
                next_state = current_state.copy()
                invalid = True
                
            # Stochastic ghost moves and their probabilities
            possible_ghosts_actions = []
            ghosts_actions_pmfs = []
            for ghost_index in range(1, self.number_of_movables):
                possible_ghost_action_list, pmf = ghost_move_pathfinding(next_state, ghost_index, self.moves, self.map, self.power)
                possible_ghosts_actions.append(possible_ghost_action_list)
                ghosts_actions_pmfs.append(pmf)

            permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
            permuted_pmfs = [list(p) for p in product(*ghosts_actions_pmfs)]
            ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]

            next_states = []
            for actions in permuted_actions:
                ghosts_state = []
                for i in range(len(actions)):
                    ghosts_state.append((actions[i][0] + next_state[1 + i][0], actions[i][1] + next_state[1 + i][1]))
                next_states.append([next_state[0]] + ghosts_state + next_state[self.number_of_movables:])
            
            next_state = choices(next_states, weights=ghosts_action_permutations_pmfs, k=1)[0]

            # Reward function
            reward = self.reward(next_state, eaten) if not invalid else self.invalid_move_reward

            terminal = is_terminal(next_state, self.number_of_movables)

            self.store_transition(current_state, action, reward, next_state, terminal)

            current_state = next_state

            # Only learn after 1/10 of the episodes passed
            #This enables the memory to initially fill with meaningless but explorative actions using high epsilon values.
            if self.episodes > self.max_episodes // 10:

                # MLP-based Q-network training
                self.sample_and_learn()

            if terminal:
                self.episodes += 1
                n_games += 1
                n_wins += int(is_win_terminal(next_state, self.number_of_movables))

                # Reset the game graphics
                if self.renderer is not None:
                    self.renderer.render(current_state, action)
                
                # Reset the game state
                current_state = self.initial_state.copy()

                if self.random_spawn:
                # Randomize next game 

                    # Candies
                    impossible_pacman_spawns = []
                    for i in range(self.number_of_movables, len(current_state)):
                        current_state[i] = int(random() < 0.5)
                        if current_state[i] == 1:
                            impossible_pacman_spawns.append(self.candies_positions[i])
                        
                    # At least one candy is 1
                    if sum(current_state[self.number_of_movables:]) == 0:
                        current_state[self.number_of_movables + int(random() * self.number_of_candies)] = 1

                    # Ghosts
                    for i in range(1, self.number_of_movables):
                        current_state[i] = choice(self.possible_positions)
                    
                    # Pacman, avoid ghosts and active candies
                    current_state[0] = choice(self.possible_positions)
                    while current_state[0] in current_state[1:self.number_of_movables] or current_state[0] in impossible_pacman_spawns:
                        current_state[0] = choice(self.possible_positions)


                # Every 10 episodes print the winrate and epsilon
                if self.logging and self.episodes % 10 == 0: 
                    print(f"Episode: {self.episodes}, winrate: {n_wins/n_games}, wins: {n_wins}, current epsilon: {self.epsilon}")
               
                # Decay epsilon linearly towards 0 at the end of the training
                self.epsilon = max(self.min_epsilon, original_epsilon - (original_epsilon / self.max_episodes) * 5 * self.episodes)    
                
                # Every 100 episodes store the Q function weights
                if self.episodes % 100 == 0:
                    self.store_Q()

                if self.episodes % 500 == 0 and self.epsilon == self.min_epsilon and self.increasing_power:
                    self.power += 1 if self.power < 10 else 10
                    print(f"\n\n{'*'*100}\n")
                    print(f"Power increased to {self.power}")
                    print(f"{'*'*100}\n\n")
                    

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


class Game:
    def __init__(self, policy_iterator, pretrained=True, tile_size=32, fps=10, power=None, logging=False, measure_performance=False, monte_carlo=False): 
        """
        Game class to run the Pacman game with the policy learned from the value iteration algorithm (requires policy_iterator instance)

        policy_iterator: policy_iterator instance, built as policy_iterator(enumerator, alpha, delta, epsilon, lose_cost, win_cost, move_cost, eat_cost, power, control_accuracy, logging)

        pretrained:     Flag to load the policy from a txt file instead of the value iterator instance

        tile_size:      Size of the tiles in the game window

        fps:            Frames per second of the game

        power:          Power parameter for the ghost moves, the higher the power the more the ghosts will try to get closer to pacman, 
                        power = 0 means the ghosts move randomly with uniform probability wrt the possible moves
                        power = 1 means the ghosts move with a probability proportional to the inverse of the manhattan distance of A* paths to pacman
                        power = 2 means the ghosts move with a probability proportional to the inverse of the squared manhattan distance of A* paths to pacman 
                        and so on...

        logging:        Flag to enable logging of the game steps

        measure_performance: Flag to enable the performance measurement of the policy

        monte_carlo:    Flag to enable the Monte Carlo simulation of the game
        """
        
        # Save the parameters from the value iterator
        self.map = policy_iterator.map
        self.possible_positions = policy_iterator.possible_positions
        self.initial_state = policy_iterator.initial_state
        self.current_state = self.initial_state.copy()
        self.number_of_movables = policy_iterator.number_of_movables
        self.candies_positions = policy_iterator.candies_positions
        self.moves = policy_iterator.moves   
        self.reward = policy_iterator.reward

        self.logging = logging
        self.measure_performance = measure_performance or monte_carlo
        self.monte_carlo = monte_carlo

        # Save the map filename
        self.map_name= policy_iterator.filename

        # Initialize the power parameter
        if power == None:
            self.power = policy_iterator.power
        else:
            self.power = power
        
        if pretrained:
            policy_iterator.load_Q()

        # Initialize the Q function
        self.Q = policy_iterator.Q

        # check if the dictionary is empty
        if not self.Q:
            raise ValueError("The Q function is not initialized. Please run the policy iteration algorithm first.")
        
        if not self.measure_performance:
            self.renderer = Renderer(policy_iterator, tile_size, fps)
        
        # At test time, account for a minimum threshold in seconds, the number of candies eaten and the number of moves
        if self.measure_performance:
            self.loop_till_loss = True
            self.logging = False
            self.min_threshold = 600 # moves
            self.max_threshold = 6000 # moves
            self.candies_eaten = 0
            self.number_of_moves = 0
            self.efficiency_ratio = 0
            self.reward_sum = 0

            self.alpha = policy_iterator.alpha
            self.epsilon = policy_iterator.epsilon
            self.gamma = policy_iterator.gamma

            self.lose_reward = policy_iterator.lose_reward
            self.win_reward = policy_iterator.win_reward
            self.move_reward = policy_iterator.move_reward
            self.eat_reward = policy_iterator.eat_reward

    def pi(self, state):
        state_tensor = encode_state(state, self.number_of_movables, self.candies_positions, self.map)
        with torch.no_grad():
            q_vals = self.Q(state_tensor)
            return q_vals.argmax().item()


    def respawn_candies(self, random_spawn=False):
        while is_win_terminal(self.current_state, self.number_of_movables):
            if len(self.candies_positions) == 1:
                self.current_state[self.number_of_movables] = 1
                new_pacman_position = choice(self.possible_positions)
                while new_pacman_position in self.current_state[1:] or new_pacman_position == self.candies_positions[self.number_of_movables]:
                    new_pacman_position = choice(self.possible_positions)
                self.current_state[0] = new_pacman_position
            else:
                for candy_index in self.candies_positions:
                    if not random_spawn:
                        # Respawn all candies a part from the last one eaten
                        if self.current_state[0] != self.candies_positions[candy_index]:
                            self.current_state[candy_index] = 1
                    else:
                        # Randomly respawn one or more candies
                        if choice([True, False]) and self.current_state[0] != self.candies_positions[candy_index]:
                            self.current_state[candy_index] = 1


    def run(self, ghost_controlled=False, loop_till_loss=False, measure_filename=""):
        running = True

        # reset measures for multiple tests
        if self.measure_performance:
            self.candies_eaten = 0
            self.number_of_moves = 0
            self.efficiency_ratio = 0
            self.reward_sum = 0

        if not (self.monte_carlo or self.measure_performance):
            # Randomize the initial positions of all ghosts
            for ghost_index in range(1, self.number_of_movables):
                new_ghost_position = choice(self.possible_positions)
                # Avoid placing a ghost on pacman's position
                while new_ghost_position == self.current_state[0]:
                    new_ghost_position = choice(self.possible_positions)
                self.current_state[ghost_index] = new_ghost_position
        
        action = 0
        is_paused = False

        if not self.measure_performance:
            # Before starting the game, display the logo 
            self.renderer.display_logo()

        while running:
            if self.measure_performance:
                if self.number_of_moves % 100 == 0:
                    if measure_filename == "": print(f"Simulated number of moves: {self.number_of_moves}")
                if self.number_of_moves == self.max_threshold:
                    self.efficiency_ratio = self.candies_eaten / self.number_of_moves
                    if measure_filename == "": print(f"Test passed - Maximum number of moves ({self.max_threshold}) reached\n\tPacman efficiency ratio: {self.efficiency_ratio}\n\tCandies eaten: {self.candies_eaten}")
                    running = False
                    continue

            key_pressed = not ghost_controlled # track whether a KEYDOWN happened

            # Draw current state every frame, so the window remains responsive 
            if not self.measure_performance:
                self.renderer.render(self.current_state, action)

                # Collect events 
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        # If any key was pressed, we consider it a turn
                        key_pressed = True
                        # If the key pressed was the pause key, we toggle the pause state
                        if event.key == pygame.K_p:
                            is_paused = not is_paused
                            if self.logging: print(f"Game paused: {is_paused}")
                
                # If paused, skip game updates and slow the loop
                if is_paused:
                    self.renderer.clock_tick(10)  # Slow loop to save CPU
                    continue

            # If no key was pressed, we do nothing and skip to next iteration 
            if not key_pressed:
                # Limit CPU usage even when not moving
                if not self.measure_performance:
                    self.renderer.clock_tick()
                continue

            # If we get here, it means a key was pressed, so we process a turn
            # Check win/lose conditions 
            if loop_till_loss:
                self.respawn_candies(random_spawn=not(self.monte_carlo or self.measure_performance))

            elif is_win_terminal(self.current_state, self.number_of_movables):
                print("Game over - You won")
                running = False
                sleep(2)
                if not self.measure_performance:
                    self.renderer.clock_tick()
                if self.logging: 
                    print(f"Wins, terminal state: {self.current_state}")

                continue

            if is_lose_terminal(self.current_state, self.number_of_movables):
                # Pacman lost before the maximum threshold
                if self.measure_performance:
                    self.efficiency_ratio = self.candies_eaten / self.number_of_moves
                    if measure_filename == "": print(f"Test failed - Pacman eaten by a ghost after {self.number_of_moves} moves\n\tPacman efficiency ratio: {self.efficiency_ratio}\n\tCandies eaten: {self.candies_eaten}")
                    running = False
                else:
                    self.renderer.clock_tick()

                print("Game over - You lost")
                running = False
                sleep(2)

                if self.logging:
                    print(f"Defeat, terminal state: {self.current_state}")

                continue

            # Move pacman according to the policy
            action = self.pi(self.current_state)

            next_state, eaten = pacman_move(self.current_state, self.moves[action], self.number_of_movables, self.candies_positions, self.map)

            invalid = False
            # If the pacman move is invalid, we keep the current state
            if next_state == False:
                next_state = self.current_state.copy()
                invalid = True

            # Account for the number of moves
            if self.measure_performance:
                self.number_of_moves += 1
                self.candies_eaten += int(eaten)

            if self.logging: 
                print(f"Pacman action: {action} triggered transition from State: {self.current_state} to State: {next_state}")

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
                        action_list, pmf = ghost_move_pathfinding(next_state, ghost_index, self.moves, self.map, self.power)
                        possible_ghosts_actions.append(action_list)
                        ghosts_actions_pmfs.append(pmf)

                    # Build permutations
                    permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
                    permuted_pmfs = [list(p) for p in product(*ghosts_actions_pmfs)]
                    ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]

                    # Sample a combination
                    ghosts_actions = choices(permuted_actions, weights=ghosts_action_permutations_pmfs, k=1)[0]

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
                if not self.measure_performance:
                    # Event handling - listen for arrow key down and up to update the fps
                    for event in events:
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                self.renderer.fps += 50
                            if event.key == pygame.K_DOWN:
                                self.renderer.fps -= 50
                            if self.renderer.fps <= 0:
                                self.renderer.fps = 1
                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                exit()

                # If ghost_controlled = False, all ghosts move stochastically
                possible_ghosts_actions = []
                ghosts_actions_pmfs = []
                for ghost_index in range(1, self.number_of_movables):
                    action_list, pmf = ghost_move_pathfinding( next_state, ghost_index, self.moves, self.map, self.power)
                    possible_ghosts_actions.append(action_list)
                    ghosts_actions_pmfs.append(pmf)
                
                # Build permutations
                permuted_actions = [list(p) for p in product(*possible_ghosts_actions)]
                permuted_pmfs = [p for p in product(*ghosts_actions_pmfs)]
                ghosts_action_permutations_pmfs = [prod(pmfs) for pmfs in permuted_pmfs]

                # Randomly choose the configuration
                ghosts_actions = choices(permuted_actions, weights=ghosts_action_permutations_pmfs, k=1)[0]

                updated_positions = []
                for i, ga in enumerate(ghosts_actions):
                    gx, gy = next_state[1 + i]
                    updated_positions.append((gx + ga[0], gy + ga[1]))

                next_state = [next_state[0]] + updated_positions + next_state[self.number_of_movables:]

                if self.logging:
                    print(f"Ghosts actions triggered transition to State: {next_state}")

            # Update current state
            self.current_state = next_state

            # Accumuate the stage cost
            if self.measure_performance:
                self.reward_sum += self.reward(self.current_state, eaten)
            else:
                self.renderer.clock_tick()


        if self.measure_performance and not self.monte_carlo:
            params = f"efficiency = {self.efficiency_ratio}, number_of_moves = {self.number_of_moves}, candies_eaten = {self.candies_eaten} - alpha = {self.alpha}, delta = {self.delta}, epsilon = {self.epsilon}, lose_cost = {self.lose_cost}, win_cost = {self.win_cost}, move_cost = {self.move_cost}, eat_cost = {self.eat_cost}, training_power = {self.training_power}, game_power = {self.power}"
            if self.number_of_moves < self.min_threshold:
                with open("./parallel_jobs/"+measure_filename+"_under_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")
            elif self.number_of_moves < self.max_threshold:
                with open("./parallel_jobs/"+measure_filename+"_between_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")
            else:
                with open("./parallel_jobs/"+measure_filename+"_over_threshold.txt", "a") as file:
                    file.write(f"{self.map_name} - {params}\n")

        elif self.monte_carlo:
            performance_params = f"efficiency = {self.efficiency_ratio}, stage_cost_sum = {self.reward_sum}"
            if self.number_of_moves < self.min_threshold:
                with open("./monte_carlo_RL/"+measure_filename+"_under_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")
            elif self.number_of_moves < self.max_threshold:
                with open("./monte_carlo_RL/"+measure_filename+"_between_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")
            else:
                with open("./monte_carlo_RL/"+measure_filename+"_over_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")

        # Quit the game once we exit the loop 
        pygame.quit()
