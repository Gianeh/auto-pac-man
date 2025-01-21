# parallel_grid_search.py
# Description: This script runs a grid search for hyperparameter tuning in parallel.

from joblib import Parallel, delayed
from dynamic_programming import States_enumerator, Value_iterator, Game
import os
from multiprocessing import Manager

game_powers = [0,2,4,6,8,10]
simulations_number = 100

enumerator = States_enumerator(map_filename="map_2", logging=True)
value_iterator = Value_iterator(enumerator, alpha=0.9, epsilon=0.01, lose_cost=200, win_cost=-500, move_cost=1, eat_cost=-1, power=2, control_accuracy=True, logging=True)
#value_iterator.run()
#value_iterator.store_policy()
#value_iterator.store_value_function()

manager = Manager()
progress_dict = manager.dict()
progress_dict["done"] = 0
progress_lock = manager.Lock() 

def run_experiment(value_iterator, simulations_number, game_power, progress_dict, progress_lock, total):
    with open("./monte_carlo/game_power_" + str(game_power) + ".log", "a") as f:
        f.write("\n\n" + "#"*170 + "\n")
        f.write(f"Running Monte Carlo simulation with map: {value_iterator.filename} and game_power = {game_power}\n")
        f.write("#"*170 + "\n\n")

    for simulation in range(1, simulations_number+1):
        with open("./monte_carlo/game_power_" + str(game_power) + ".log", "a") as f:
            f.write(f"\nRunning simulation: {simulation}\n")
        pacmam_game = Game(value_iterator, pretrained=True, tile_size=50, fps=1000, power=game_power, measure_performance=True, monte_carlo=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(game_power))

    # Increment the global progress
    with progress_lock:
        progress_dict["done"] += 1
        done = progress_dict["done"]
    print(f"[PID {os.getpid()}] Completed iteration {done} / {total}.")


if __name__ == "__main__":
    os.makedirs("monte_carlo", exist_ok=True)
    param_grid = []
    for game_power in game_powers:
        param_grid.append((value_iterator, simulations_number, game_power))

    total = len(param_grid)

    # Notice we pass the manager objects to each job
    Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            *params, progress_dict, progress_lock, total
        ) for params in param_grid
    )

    print("\nAll experiments finished!")


    