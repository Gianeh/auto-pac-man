# Auto Pac-Man – Dynamic Programming Branch

This branch implements a dynamic programming (DP) solution for solving Pac-Man via value iteration. The code in this branch enumerates game states from a map, performs value iteration to compute an optimal policy, and then runs a Pygame simulation of Pac-Man using that policy.

> **Disclaimer:**  
> All the codebase is experimental and no longer deeply maintained. Feel free to experiment and propose edits!

---

## Overview

The DP branch is structured around three main components:

- **States Enumeration:**  
  The `States_enumerator` class loads a map (from an image or text file), identifies free positions, and generates all possible game states. A state consists of the Pac-Man position, positions of ghosts, and the status of candies (present or eaten).

- **Value Iteration Algorithm:**  
  The `Value_iterator` class implements the value iteration algorithm. It iterates over the enumerated state space, updating a value function based on the cost of actions (moving, eating a candy, being caught by a ghost, etc.). It eventually converges on an optimal policy that minimizes expected costs.  
  Key hyperparameters include:
  - **alpha:** Discount factor.
  - **delta/epsilon:** Convergence threshold/accuracy control.
  - **Cost parameters:** `lose_cost`, `win_cost`, `move_cost`, and `eat_cost`.
  - **Power parameter:** Influences ghost movement behavior.

- **Game Simulation:**  
  The `Game` class uses Pygame to run the Pac-Man simulation using the policy learned from value iteration. This simulation supports options such as ghost control and looping until Pac-Man loses.

---

## Files and Code Structure

- **Main Script:**  
  The entry point for the DP branch is a Python script that:
  - Parses command-line arguments for map selection, training options, and various hyperparameters.
  - Creates a `States_enumerator` instance to load and enumerate game states.
  - Initializes a `Value_iterator` to run the value iteration algorithm.
  - Optionally trains the model and stores the policy and value function.
  - Launches a Pygame-based game through the `Game` class.
  
- **Dynamic Programming Module:**  
  Contains the core classes:
  - `States_enumerator` for map loading and state enumeration.
  - `Value_iterator` for the DP-based value iteration.
  - `Game` for running the simulation.
  
- **Helper Functions:**  
  Additional functions (e.g., for checking terminal states, computing move costs, ghost movement strategies) assist in state updates and cost evaluations.

---

## Usage

### Requirements

- Python 3.x
- Pygame
- Pillow (PIL)
- NumPy

Ensure these packages are installed (e.g., via `pip install pygame Pillow numpy`).

### Command-Line Arguments

Run the script from the command line with various parameters. For example:

```bash
python main.py --filename map_1 --train --logging \
               --alpha 0.9 --delta 0.01 --epsilon 1.0 \
               --lose_cost 200 --win_cost -500 --move_cost 1 --eat_cost -1 \
               --power 2 --ghost_controlled --loop_till_loss
```

- **Map Selection:**  
  `--filename` selects the map to load (without file extension).  
  By default, the map is loaded from a PNG image (`./maps/<filename>.png`); alternatively, a text file can be used.

- **Training Options:**  
  Use `--train` to perform value iteration training before running the game.  
  Enable `--logging` for detailed output during training.

- **Hyperparameters:**  
  Adjust the discount factor (`--alpha`), convergence thresholds (`--delta` and `--epsilon`), and cost parameters (`--lose_cost`, `--win_cost`, `--move_cost`, `--eat_cost`) to experiment with different DP settings.

- **Game Options:**  
  Use `--ghost_controlled` if you want the ghosts’ moves to be controlled by the model rather than random.  
  Use `--loop_till_loss` to have the game run continuously until Pac-Man loses.

### Running the DP Code

1. **Training:**  
   If training is enabled (`--train`), the script will:
   - Enumerate all possible game states.
   - Run the value iteration algorithm until the value function converges.
   - Store the computed policy and value function in the `./policies` and `./value_functions` directories respectively.

2. **Simulation:**  
   Once trained (or when using a pre-trained policy), the game is launched using the `Game` class. The simulation displays the game board at the specified tile size and frame rate.

---

## Customization and Future Work

- **State Space Enhancements:**  
  The state enumeration currently considers free positions for Pac-Man, ghosts, and candies. Future work may optimize this process to handle larger maps or additional game dynamics.

- **Algorithm Tuning:**  
  Experiment with different values for hyperparameters and cost functions to refine the policy’s performance. The "power" parameter adjusts how aggressively ghosts pursue Pac-Man.

- **Integration with Other Branches:**  
  This DP branch is part of a larger project exploring multiple solution approaches (DP, RL, Deep RL). For alternative implementations, refer to the corresponding branches.

---

## Acknowledgments

This project is the outcome of collaborative academic work and represents initial efforts in exploring dynamic programming strategies for game AI.

Insipiration was taken mainly from:
- Berkely AI Materials on Pac-man projects: https://ai.berkeley.edu/project_overview.html
- Tycho van der Ouderaa adapted Deep RL version of the above: https://github.com/tychovdo/PacmanDQN

---
