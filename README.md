# Auto Pac-Man – Deep Q-Learning

Deep Q-learning solution for Pac-Man. This branch hosts solutions using both a Convolutional Neural Network (CNN) and a Multilayer Perceptron (MLP). While the CNN version has produced valid results on all maps, the MLP version — though included for experimentation — is not well-suited for the complex ClassicGrid map (from [Berkeley AI Pac-Man Projects](https://ai.berkeley.edu/project_overview.html)) and requires additional tuning to perform reliably even on the simpler test maps used in the DP and RL branches.

> **Disclaimer:**  
> All code in this branch is experimental and no longer deeply maintained. Feel free to experiment and propose edits!

---

## Overview

![Neural RL Demo](https://github.com/Gianeh/auto-pac-man/blob/neural_rl/dql_demo.gif "Game demo")

The deep Q-learning branch is built around several key components:

- **State Encoding:**  
  The game state is encoded as a multi-channel 2D map. Each channel represents a different element (floor, wall, candy, Pac-Man, ghost) and serves as the input to the neural network.

- **Neural Network Architectures:**  
  - **CNN-Based Q Function:**  
    The `CNNNetwork` class defines a convolutional network that processes the encoded state and outputs Q values for all possible actions. This version has produced consistent and valid results across all provided maps.
  - **MLP-Based Q Function:**  
    An alternative MLP network is also provided. However, initial experiments indicate that the MLP struggles with the complexity of the ClassicGrid map and even the test maps used in the DP and RL branches. This implementation requires further tuning to be competitive.

- **Deep Q-Learning with Experience Replay:**  
  The `Neural_Policy_iterator` class implements deep Q-learning using experience replay, an epsilon-greedy exploration strategy, and periodic updates to a target network (using Double DQN). Additional features include:
  - **Adaptive Exploration:** Epsilon decays exponentially from its initial value to a specified minimum.
  - **Replay Buffer:** Transitions are stored and randomly sampled to improve training stability.
  - **Target Network Updates:** The target network is periodically synchronized with the primary Q network.

- **Rendering and Simulation:**  
  The `Renderer` class (using Pygame) visualizes the training process in real time. The `Game` class launches the Pac-Man simulation using the learned policy.

---

## Files and Code Structure

- **Main Script:**  
  The entry point (`main.py`) sets up the environment:
  - Parses command-line arguments for map selection, training options, and various hyperparameters.
  - Initializes the game state using the shared `State_initializer`.
  - Instantiates the `Neural_Policy_iterator` (which supports both CNN and MLP versions) for deep Q-learning.
  - Optionally renders the training process via the `Renderer` class.
  - Launches the game simulation using the `Game` class with the trained network.

- **Neural RL Module:**  
  Contains the core components:
  - `CNNNetwork`: Defines the CNN architecture for Q-value approximation.
  - `Neural_Policy_iterator`: Implements the deep Q-learning training loop, including experience replay and target network updates.
  - Alternative MLP-based network implementation (experimental).
  - `Renderer` and `Game`: Handle visualization and simulation.

- **Helper Functions:**  
  Shared utility functions (e.g., `is_terminal`, `pacman_move`, `ghost_move_pathfinding`) support state transitions, reward computation, and ghost movement.

---

## Usage

### Requirements

- Python 3.x
- Pygame
- Pillow (PIL)
- NumPy
- PyTorch

Install the required packages using:

```bash
pip install pygame Pillow numpy torch
```

### Command-Line Arguments

Run the script from the command line with various parameters. For example:

```bash
python main.py --filename map_1 --load_txt --train --load_checkpoint --max_episodes 500000 --render_training --random_spawn --logging \
               --alpha 0.05 --gamma 0.95 --epsilon 1.0 --min_epsilon 0.05 --power 2 \
               --eat_reward 20 --move_reward -5 --lose_reward -500 --win_reward 100 \
               --game_power 2 --loop_till_loss --ghost_controlled
```

- **Map Selection:**  
  `--filename` specifies the map to load (without file extension). By default, the map is loaded from a PNG image (located in `./maps/<filename>.png`), but you can opt to load from a text file using `--load_txt`.

- **Training Options:**  
  - Use `--train` to initiate deep Q-learning training.
  - Use `--load_checkpoint` to load previously stored CNN weights.
  - Enable `--render_training` to visualize the training process.

- **Hyperparameters:**  
  Adjust the learning rate (`--alpha`), discount factor (`--gamma`), exploration parameters (`--epsilon` and `--min_epsilon`), and reward values (`--eat_reward`, `--move_reward`, `--lose_reward`, `--win_reward`) to experiment with different configurations.

- **Game Options:**  
  - Use `--random_spawn` to randomize the starting positions of Pac-Man, ghosts, and candies each episode.
  - Use `--ghost_controlled` to have ghost movements influenced by the model.
  - Use `--loop_till_loss` to run the game continuously until Pac-Man loses.
  - The `--game_power` parameter sets the ghost movement power for the simulation (defaulting to the value of `--power` if not specified).

---

## Customization and Future Work

- **Network Architecture:**  
  While the CNN version has shown robust performance, the MLP implementation needs further refinement to handle more complex maps like ClassicGrid effectively.

- **Experience Replay Enhancements:**  
  Future improvements might include prioritized experience replay or larger replay buffers to enhance sample efficiency.

- **Hyperparameter Tuning:**  
  Experiment with different epsilon decay rates, learning rates, and target network update frequencies to optimize training dynamics.

- **Integration with Other Branches:**  
  This branch complements the DP and traditional RL branches. For alternative approaches, refer to their respective branches.

---

## Acknowledgments

This project is the result of collaborative academic work exploring various AI strategies for game control. Inspiration was drawn from:
- Berkeley AI Materials on Pac-Man projects: [Berkeley AI Pac-Man Projects](https://ai.berkeley.edu/project_overview.html)
- Tycho van der Ouderaa’s adaptation of Deep RL for Pac-Man: [PacmanDQN](https://github.com/tychovdo/PacmanDQN)


---
