# Classes for State counting and reinforcement learning training with CNN
from math import prod, exp
from itertools import product
from random import choices, choice, random, sample
from time import sleep
import pygame
import os
from helper_functions import is_terminal, pacman_move, is_win_terminal, is_lose_terminal, ghost_move_pathfinding, Renderer
import copy

# classes for neural network class
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN class for the Q function
class CNNNetwork(nn.Module):
    def __init__(self, input_channels, action_dim, device):
        super(CNNNetwork, self).__init__()
        self.device = device  # Store the device
        # Define a simple CNN with convolutional and fully connected layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)  # Assuming input map is 7x7 after pooling
        self.fc2 = nn.Linear(256, action_dim)
        self.to(self.device)  # Move the model to the specified device

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (7, 7))  # Ensure consistent size
        x = torch.flatten(x, start_dim=1)      # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Updated state encoding to create a 2D map representation
def encode_state_as_map(state, number_of_movables, candies_positions, encoded_map):

    # Place Pacman
    pacman_x, pacman_y = state[0]
    encoded_map[3, pacman_y, pacman_x] = 1

    # Place Ghosts
    for i in range(1, number_of_movables):
        ghost_x, ghost_y = state[i]
        encoded_map[4, ghost_y, ghost_x] = 1

    # Place Candies
    for candy_idx in range(number_of_movables, len(state)):
        candy_state = state[candy_idx]
        candy_x, candy_y = candies_positions[candy_idx]
        if candy_state == 1:  # Candy is active
            encoded_map[2, candy_y, candy_x] = 1

    return encoded_map

# Generalized Policy iteration algorithm using CNN-based Q-Learning
class Neural_Policy_iterator:
    def __init__(self, initializer, renderer=None, max_episodes=1000, pretrained=False, alpha=1e-3, gamma=9e-1, epsilon=1e0, min_epsilon=5e-2, lose_reward=-1e3, win_reward=5e3, move_reward=-1, eat_reward=1e1, power=10, increasing_power=False, random_spawn=False, logging=False):
        """
        Generalized Policy Iteration algorithm using Deep Q-Learning with a CNN (requires State_initializer instance)

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
        self.map_shape = (len(self.map), len(self.map[0]))  # Height, Width
        self.initial_state = initializer.initial_state
        self.number_of_movables = initializer.number_of_movables
        self.number_of_ghosts = initializer.number_of_ghosts
        self.number_of_candies = initializer.number_of_candies
        self.candies_positions = initializer.candies_positions
        self.possible_positions = initializer.possible_positions
        self.filename = initializer.filename

        # Save the device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.random_spawn = random_spawn
        self.power = power
        self.increasing_power = increasing_power

        # save policy iteration hyperparameters
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon

        # Replay buffer for experience replay
        self.replay_buffer = []
        self.replay_capacity = 100000 

        # How many transitions to sample from the replay buffer for each training step of the Q-network
        self.batch_size = 128

        # Number of trainig steps (of the Q-network) before updating the target network
        self.update_target_steps = 5000

        # Counter for the number of learning steps of the Q-network
        self.learn_step_counter = 0

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

        # Initialize CNN-based Q function and target network
        self.Q = CNNNetwork(input_channels=5, action_dim=len(self.moves), device=self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)

        if pretrained:
            self.load_Q()

        # Renderer object to visualize the game
        self.renderer = renderer

        self.logging = logging

        # Episode counter
        self.episodes = 0

        # Initialize the map encoding with walls
        self.encoded_map = torch.zeros((5, self.map_shape[0], self.map_shape[1]), dtype=torch.float32)  # 5 channels: floor, wall, candy, pacman, ghost
        for y in range(self.map_shape[0]):
            for x in range(self.map_shape[1]):
                cell_value = self.map[y][x]
                if cell_value == 1:  # Wall
                    self.encoded_map[1, y, x] = 1
                else: # Floor or possible position
                    self.encoded_map[0, y, x] = 1

    
    # Store the Q function weights in a file
    def store_Q(self):
        os.makedirs("./cnn_weights", exist_ok=True)
        torch.save(self.Q.state_dict(), f"./cnn_weights/{self.filename}_Q.pt")   
    # Load the Q function weights from a file
    def load_Q(self):
        if os.path.exists(f"./cnn_weights/{self.filename}_Q.pt"):
            self.Q.load_state_dict(torch.load(f"./cnn_weights/{self.filename}_Q.pt", weights_only=True))
            print(f"Pretrained weights loaded from ./cnn_weights/{self.filename}_Q.pt")
        else:
            print(f"No pretrained weights found at ./cnn_weights/{self.filename}_Q.pt")

    def reward(self, state, eaten):
        r = 0
        # if pacman is eaten by a ghost
        if state[0] in state[1:self.number_of_movables]:
            r += self.lose_reward
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
            
            # If the ghost is in any of those neighboring cells, apply a discounted lose reward
            if (ghost_x, ghost_y) in neighbors:
                r += self.lose_reward/5
            
        # if pacman ate a candy
        if eaten:
            if sum(state[self.number_of_movables:]) == 0:
                r += self.win_reward
            r += self.eat_reward
        # otherwise
        return r + self.move_reward
    
    def encode_state(self, state):
        self.encoded_map[2, :, :] = 0  # Reset candies
        self.encoded_map[3, :, :] = 0  # Reset pacman
        self.encoded_map[4, :, :] = 0  # Reset ghosts
        state_tensor = encode_state_as_map(state, self.number_of_movables, self.candies_positions, self.encoded_map)
        return state_tensor.to(self.device)  # Move to GPU
    
    def store_transition(self, state, action, reward, next_state, terminal):
        # Assuming `state` is a tensor that should be moved to the device
        state = self.encode_state(state)
        next_state = self.encode_state(next_state)
        self.replay_buffer.append((state, action, reward, next_state, terminal))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

    def pi(self, state):
        if random() < self.epsilon:
            return choice(list(self.moves.keys()))
        else:
            state_tensor = self.encode_state(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_vals = self.Q(state_tensor)
                return q_vals.argmax().item()

    def sample_and_learn(self):
        # Don’t update if not enough transitions yet
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a random batch from replay buffer
        transitions = sample(self.replay_buffer, self.batch_size)

        # Convert them into batches of tensors
        current_states, actions, rewards, next_states, terminals = zip(*transitions)

        # State Tensor: every 3d tensor is a current state of the transition
        current_states_tensor = torch.stack([state for state in current_states]).to(self.device)
        # Next State Tensor: every 3d tensor is a next state of the transition
        next_states_tensor = torch.stack([next_state for next_state in next_states]).to(self.device)
        # Actions Tensor: every row is the action taken in the transition
        actions_tensor = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        # Rewards Tensor: every row is the reward of the transition
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        # Terminal Tensor: every row is the terminal flag of the next state of the transition
        terminals_tensor = torch.tensor(terminals, dtype=torch.int8).view(-1, 1).to(self.device)

        # Get current Q-values from main network
        q_values = self.Q(current_states_tensor)
        # Gather the Q-value for the chosen action
        q_a = q_values.gather(1, actions_tensor)

        with torch.no_grad():
            """ Classic DQN
            q_next = self.Q_target(next_states_tensor)
            q_next_max = q_next.max(dim=1, keepdim=True)[0]
            q_target = rewards_tensor + (1 - terminals_tensor) * self.gamma * q_next_max
            """
            # Using Double DQN
            q_next = self.Q(next_states_tensor)
            q_next_argmax = q_next.argmax(dim=1, keepdim=True)
            q_next_max = self.Q_target(next_states_tensor).gather(1, q_next_argmax)
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
        n_steps = 0     # Number of steps in the current game (used to control the CNN training frequency)

        # Main loop of the policy iteration
        while self.episodes <= self.max_episodes:

            # Count the number of steps in the current game
            n_steps += 1

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

            # First ghost moves according to power setting
            first_ghost_action_list, first_ghost_pmf = ghost_move_pathfinding(next_state, 1, self.moves, self.map, self.power)
            possible_ghosts_actions.append(first_ghost_action_list)
            ghosts_actions_pmfs.append(first_ghost_pmf)

            # Other ghosts move randomly
            for ghost_index in range(2, self.number_of_movables):
                possible_ghost_action_list, pmf = ghost_move_pathfinding(next_state, ghost_index, self.moves, self.map, power=0)
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
            reward = self.reward(next_state, eaten) if not invalid else self.lose_reward/4

            terminal = is_terminal(next_state, self.number_of_movables)

            self.store_transition(current_state, action, reward, next_state, terminal)

            
            n_wins += int(is_win_terminal(next_state, self.number_of_movables))

            current_state = next_state

            # Only learn after 1/10 of the episodes passed
            #This enables the memory to initially fill with meaningless but explorative actions using high epsilon values.
            if self.episodes > self.max_episodes // 10 and n_steps % 4 == 0:
                
                # CNN-based Q-Network training
                self.sample_and_learn()
                # Decay epsilon exponentially towards min_epsilon#self.epsilon = max(self.min_epsilon, original_epsilon - (original_epsilon / self.max_episodes) * 2.5 * self.episodes)
                self.epsilon = self.min_epsilon + ((original_epsilon - self.min_epsilon) * exp(-0.001 * (self.episodes - self.max_episodes // 10)))
                
            if terminal:
                n_steps = 0
                self.episodes += 1
                n_games += 1

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
                    print(f"Episode: {self.episodes}, winrate: {n_wins/n_games}, n_wins: {n_wins}, current epsilon: {self.epsilon}")
               
                
                # Every 100 episodes store the Q function weights
                if self.episodes % 100 == 0:
                    self.store_Q()

                
                if self.episodes % 500 == 0 and self.epsilon <= self.min_epsilon and self.increasing_power:
                    self.power += 1 if self.power < 10 else 10
                    if self.logging:
                        print(f"\n\n{'*'*100}\n")
                        print(f"Power increased to {self.power}")
                        print(f"{'*'*100}\n\n")
                

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
        self.map_shape = policy_iterator.map_shape
        self.encoded_map = policy_iterator.encoded_map
        self.device = policy_iterator.device

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
    
        if not self.measure_performance:
            self.renderer = Renderer(policy_iterator, tile_size, fps)
        
        # At test time, account for a minimum threshold in steps, the number of candies eaten and the number of moves
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
    

    def encode_state(self, state):
        self.encoded_map[2, :, :] = 0  # Reset candies
        self.encoded_map[3, :, :] = 0  # Reset pacman
        self.encoded_map[4, :, :] = 0  # Reset ghosts
        state_tensor = encode_state_as_map(state, self.number_of_movables, self.candies_positions, self.encoded_map)
        return state_tensor.to(self.device)  # Move to GPU


    def pi(self, state):
        state_tensor = self.encode_state(state).unsqueeze(0)  # Add batch dimension
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


    def run(self, ghost_controlled=False, loop_till_loss=False, random_ghosts_spawn=False, measure_filename=""):
        running = True

        # reset measures for multiple tests
        if self.measure_performance:
            self.candies_eaten = 0
            self.number_of_moves = 0
            self.efficiency_ratio = 0
            self.reward_sum = 0

        if random_ghosts_spawn:
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

            # If the pacman move is invalid, we keep the current state
            if next_state == False:
                next_state = self.current_state.copy()

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
                        action_list, pmf = ghost_move_pathfinding(next_state, ghost_index, self.moves, self.map, power=0)
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

                first_ghost_action_list, first_ghost_pmf = ghost_move_pathfinding(next_state, 1, self.moves, self.map, self.power)
                possible_ghosts_actions.append(first_ghost_action_list)
                ghosts_actions_pmfs.append(first_ghost_pmf)

                for ghost_index in range(2, self.number_of_movables):
                    action_list, pmf = ghost_move_pathfinding( next_state, ghost_index, self.moves, self.map, power=0)
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


        if self.monte_carlo:
            performance_params = f"efficiency = {self.efficiency_ratio}, stage_cost_sum = {self.reward_sum}"
            if self.number_of_moves < self.min_threshold:
                with open("./monte_carlo_cnn/"+measure_filename+"_under_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")
            elif self.number_of_moves < self.max_threshold:
                with open("./monte_carlo_cnn/"+measure_filename+"_between_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")
            else:
                with open("./monte_carlo_cnn/"+measure_filename+"_over_threshold.txt", "a") as file:
                    file.write(f"{performance_params}\n")

        # Quit the game once we exit the loop 
        pygame.quit()