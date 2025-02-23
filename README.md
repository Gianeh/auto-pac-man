# Auto Pac-Man – Reinforcement Learning Branch

This branch implements a reinforcement learning (RL) solution for solving Pac-Man. Instead of relying on full state enumeration as in the DP branch, this RL approach uses a policy iteration algorithm based on Q-learning. The agent learns an optimal policy by interacting with the environment, updating its Q-function over many episodes, and then running a Pygame simulation using the learned policy.

> **Disclaimer:**  
> All code in this branch is experimental and no longer deeply maintained. Feel free to experiment and propose edits!

---

## Overview

![Alt text](https://github.com/Gianeh/auto-pac-man/blob/rl/rl_demo.gif "Game demo --loop_till_loss")

The RL branch is structured around several core components:

- **State Initialization:**  
  The `State_initializer` class loads a map (from an image or text file) and sets up the initial game state. This includes determining the positions for Pac-Man, ghosts, and candies, as well as computing all available free positions.

- **Policy Iteration with Q-Learning:**  
  The `Policy_iterator` class implements the RL algorithm. Key aspects include:
  - **Q-Function Updates:** Learning from interactions with the environment using a reward signal.
  - **Exploration vs. Exploitation:** An epsilon-greedy strategy is used to balance exploring new actions and exploiting learned ones.
  - **Epsilon Decay:** The exploration rate decays exponentially over episodes toward a minimum value.
  - **Hyperparameters:**  
    - **alpha:** Learning rate.
    - **gamma:** Discount factor.
    - **epsilon/min_epsilon:** Exploration parameters.
    - **Reward values:** `eat_reward`, `move_reward`, `lose_reward`, and `win_reward`.

- **Rendering and Game Simulation:**  
  The `Renderer` class provides real-time visualization during training via Pygame, allowing adjustments such as frame rate changes and pause functionality. The `Game` class launches the Pac-Man simulation using the learned policy, with options for ghost control and continuous play until Pac-Man loses.

---

## Files and Code Structure

- **Main Script:**  
  The RL branch’s entry point is a Python script that:
  - Parses command-line arguments for map selection, training options, checkpoint loading, and various hyperparameters.
  - Creates a `State_initializer` instance to load and set up the initial state.
  - Optionally renders the training process via the `Renderer` class.
  - Initializes a `Policy_iterator` to run the Q-learning algorithm, update the Q-function, and store checkpoints.
  - Launches the game simulation using the `Game` class with the learned policy.

- **Reinforcement Learning Module:**  
  Contains the core classes:
  - `State_initializer` for map loading and initial state creation.
  - `Policy_iterator` for the Q-learning based policy iteration.
  - `Renderer` for real-time visualization of training.
  - `Game` for running the final Pac-Man simulation.

- **Helper Functions:**  
  Additional functions (e.g., `is_terminal`, `pacman_move`, and `ghost_move_pathfinding`) support state transitions, reward computations, and ghost movement strategies.

---

## Usage

### Requirements

- Python 3.x
- Pygame
- Pillow (PIL)
- NumPy

Install the required packages using:

```bash
pip install pygame Pillow numpy
```

### Command-Line Arguments

Run the script from the command line with various parameters. For example:

```bash
python main.py --filename map_1 --train --load_checkpoint --max_episodes 500000 --render_training --random_spawn --logging \
               --alpha 0.05 --gamma 0.95 --epsilon 1.0 --min_epsilon 0.05 --power 2 \
               --eat_reward 20 --move_reward -5 --lose_reward -500 --win_reward 100 \
               --game_power 2 --loop_till_loss --ghost_controlled
```

- **Map Selection:**  
  `--filename` selects the map to load (without file extension). By default, the map is loaded from a PNG image (`./maps/<filename>.png`); a text file can also be used.

- **Training Options:**  
  - Use `--train` to initiate RL training.
  - Use `--load_checkpoint` to load a previously stored Q-function.
  - Enable `--render_training` to visualize the training process.

- **Hyperparameters:**  
  Adjust the learning rate (`--alpha`), discount factor (`--gamma`), exploration parameters (`--epsilon` and `--min_epsilon`), and reward values (`--eat_reward`, `--move_reward`, `--lose_reward`, `--win_reward`) to experiment with different configurations.

- **Game Options:**  
  - Use `--random_spawn` to randomize the starting positions of Pac-Man, ghosts, and candies each episode.
  - Use `--ghost_controlled` to allow ghost movements to be influenced by the model.
  - Use `--loop_till_loss` to run the game continuously until Pac-Man loses.

### Running the RL Code

1. **Training:**  
   When training is enabled (`--train`), the script will:
   - Initialize the game state using `State_initializer`.
   - Run the policy iteration algorithm for the specified number of episodes.
   - Update the Q-function using an epsilon-greedy strategy and exponential decay.
   - Optionally render the training process and store Q-function checkpoints periodically in the `./Q_tables` directory.

2. **Simulation:**  
   After training (or when using a pre-trained Q-function), the game is launched using the `Game` class. The simulation displays the game board and applies the learned policy to control Pac-Man.

---

## Customization and Future Work

- **State Space Exploration:**  
  Future improvements might include more dynamic spawning of game elements and additional state features to enhance learning efficiency.

- **Algorithm Tuning:**  
  Experiment with different hyperparameters and reward structures to fine-tune policy performance. Adjusting the exploration/exploitation balance via the epsilon decay schedule is a key area for optimization.

- **Integration with Other Branches:**  
  This RL branch complements the dynamic programming (DP) and deep reinforcement learning (Deep RL) branches. For alternative implementations, please refer to the corresponding branches.

---

## Acknowledgments

This project is the result of collaborative academic efforts exploring various AI strategies for game control. Inspiration was drawn from:
- Berkeley AI Materials on Pac-Man projects: [Berkeley AI Pac-Man Projects](https://ai.berkeley.edu/project_overview.html)
- Tycho van der Ouderaa’s adaptation of Deep RL for Pac-Man: [PacmanDQN](https://github.com/tychovdo/PacmanDQN)

---

Feel free to experiment and propose edits as you explore reinforcement learning approaches to solving Pac-Man!

