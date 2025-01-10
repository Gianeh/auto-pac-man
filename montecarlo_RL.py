# parallel_grid_search.py
# Description: This script runs a grid search for hyperparameter tuning in parallel.
'''
from joblib import Parallel, delayed
from reinforcement_learning import State_initializer, Policy_iterator, Game
import os
from multiprocessing import Manager

game_powers = [0,2,4,6,8,10]
simulations_number = 100

state_initializer = State_initializer(map_filename="ez_map", logging=True)
policy_iterator = Policy_iterator(state_initializer, max_episodes=10000, renderer=None, logging=True, power=10)
policy_iterator.run()
policy_iterator.store_Q()

manager = Manager()
progress_dict = manager.dict()
progress_dict["done"] = 0
progress_lock = manager.Lock() 

# create monte_carlo directory if it does not exist
os.makedirs("monte_carlo_RL", exist_ok=True)


def run_experiment(policy_iterator, simulations_number, game_power, progress_dict, progress_lock, total):
    with open("./monte_carlo_RL/game_power_" + str(game_power) + ".log", "a") as f:
        f.write("\n\n" + "#"*170 + "\n")
        f.write(f"Running Monte Carlo simulation with map: {policy_iterator.filename} and game_power = {game_power}\n")
        f.write("#"*170 + "\n\n")

    for simulation in range(1, simulations_number+1):
        with open("./monte_carloRL/game_power_" + str(game_power) + ".log", "a") as f:
            f.write(f"\nRunning simulation: {simulation}\n")
        pacmam_game = Game(policy_iterator, pretrained=True, tile_size=50, fps=1000, power=game_power, measure_performance=True, monte_carlo=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(game_power))

    # Increment the global progress
    with progress_lock:
        progress_dict["done"] += 1
        done = progress_dict["done"]
    print(f"[PID {os.getpid()}] Completed iteration {done} / {total}.")


if __name__ == "__main__":
    param_grid = []
    for game_power in game_powers:
        param_grid.append((policy_iterator, simulations_number, game_power))

    total = len(param_grid)

    # Notice we pass the manager objects to each job
    Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            *params, progress_dict, progress_lock, total
        ) for params in param_grid
    )

    print("\nAll experiments finished!")
'''

# parallel_grid_search.py
# Description: This script runs a grid search for hyperparameter tuning in parallel.

from joblib import Parallel, delayed
from reinforcement_learning import State_initializer, Policy_iterator, Game
import os
from multiprocessing import Manager

game_powers = [0, 2, 4, 6, 8, 10]
simulations_number = 100

def run_experiment(policy_iterator, simulations_number, game_power, progress_dict, progress_lock, total):
    log_dir = "./monte_carlo_RL/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

    log_path = os.path.join(log_dir, f"game_power_{game_power}.log")
    with open(log_path, "a") as f:
        f.write("\n\n" + "#" * 170 + "\n")
        f.write(f"Running Monte Carlo simulation with map: {policy_iterator.filename} and game_power = {game_power}\n")
        f.write("#" * 170 + "\n\n")

    for simulation in range(1, simulations_number + 1):
        with open(log_path, "a") as f:
            f.write(f"\nRunning simulation: {simulation}\n")
        pacmam_game = Game(policy_iterator, pretrained=True, tile_size=50, fps=1000, power=game_power, measure_performance=True, monte_carlo=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(game_power))

    # Increment the global progress
    with progress_lock:
        progress_dict["done"] += 1
        done = progress_dict["done"]
    print(f"[PID {os.getpid()}] Completed iteration {done} / {total}.")


if __name__ == "__main__":
    state_initializer = State_initializer(map_filename="ez_map", logging=True)
    policy_iterator = Policy_iterator(state_initializer, max_episodes=1000000, renderer=None, logging=True, power=10)
    policy_iterator.run()
    policy_iterator.store_Q()

    manager = Manager()
    progress_dict = manager.dict()
    progress_dict["done"] = 0
    progress_lock = manager.Lock()

    # Parameter grid
    param_grid = [(policy_iterator, simulations_number, game_power) for game_power in game_powers]
    total = len(param_grid)

    # Parallel execution
    Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            *params, progress_dict, progress_lock, total
        ) for params in param_grid
    )

    print("\nAll experiments finished!")
