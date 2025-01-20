# parallel_grid_search.py
# Description: This script runs a grid search for hyperparameter tuning in parallel.

from joblib import Parallel, delayed
from reinforcement_learning import State_initializer, Policy_iterator, Game
import os
from multiprocessing import Manager

maps = ["ez_map"]
epsilons = [1, 0.1]
gammas = [0.9]
alphas = [0.7, 0.9]
lose_costs = [10000, 20000, 50000]
win_costs = [-10000, -20000, -50000]
move_costs = [10, 100]
eat_costs = [-10, -100]
training_powers = [0, 10]
game_powers = [0,2,4,6,8,10]

print(f"\033[91mStarting a Grid search for hyperparameter tuning\033[0m\nTotal number of iterations: {len(maps)*len(epsilons)*len(gammas)*len(alphas)*len(lose_costs)*len(win_costs)*len(move_costs)*len(eat_costs)*len(training_powers)}\nEach training runs {len(game_powers)} games\n")

manager = Manager()
progress_dict = manager.dict()
progress_dict["done"] = 0
progress_lock = manager.Lock()  # Manager-created Lock (picklable)

def run_experiment(map_filename, epsilon, gamma, alpha, lose_cost, win_cost, move_cost, eat_cost, training_power, game_powers, progress_dict, progress_lock, total):
    with open("./parallel_jobs/pid_" + str(os.getpid()) + ".log", "a") as f:
        f.write("\n\n" + "#"*170 + "\n")
        f.write(f"Running training with map: {map_filename}, epsilon: {epsilon}, gamma: {gamma}, alpha: {alpha}, lose_cost: {lose_cost}, win_cost: {win_cost}, move_cost: {move_cost}, eat_cost: {eat_cost}, training_power: {training_power}\n")
        f.write("#"*170 + "\n\n")

    enumerator = State_initializer(map_filename=map_filename, logging=False)
    policy_iterator = Policy_iterator(enumerator, alpha=alpha, epsilon=epsilon, gamma=gamma, lose_cost=lose_cost, win_cost=win_cost, move_cost=move_cost, eat_cost=eat_cost, power=training_power, control_accuracy=True, logging=False)
    policy_iterator.run()

    for game_power in game_powers:
        with open("./parallel_jobs/pid_" + str(os.getpid()) + ".log", "a") as f:
            f.write(f"\nRunning game with game power: {game_power}\n")
        pacmam_game = Game(policy_iterator, pretrained=False, tile_size=50, fps=1000, power=game_power, measure_performance=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(os.getpid()))

    # Increment the global progress
    with progress_lock:
        progress_dict["done"] += 1
        done = progress_dict["done"]
    print(f"[PID {os.getpid()}] Completed iteration {done} / {total}.")

if __name__ == "__main__":
    param_grid = []
    for map_filename in maps:
        for epsilon in epsilons:
            for gamma in gammas:
                for alpha in alphas:
                    for lose_cost in lose_costs:
                        for win_cost in win_costs:
                            for move_cost in move_costs:
                                for eat_cost in eat_costs:
                                    for training_power in training_powers:
                                        param_grid.append((map_filename, epsilon, gamma, alpha, lose_cost, win_cost, move_cost, eat_cost, training_power, game_powers))

    total = len(param_grid)

    # Notice we pass the manager objects to each job
    Parallel(n_jobs=-1)(
        delayed(run_experiment)(
            *params, progress_dict, progress_lock, total
        ) for params in param_grid
    )

    print("\nAll experiments finished!")
