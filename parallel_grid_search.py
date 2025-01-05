# Description: This script runs a grid search for hyperparameter tuning in parallel.
from joblib import Parallel, delayed
from dynamic_programming import States_enumerator, Value_iterator, Game
import os

maps = ["ez_map"]
epsilons = [1, 0.1]
alphas = [0.7, 0.9]
lose_costs = [10000, 20000, 50000]
win_costs = [-10000, -20000, -50000]
move_costs = [1, 10, 100]
eat_costs = [-10, -100]
training_powers = [0, 10]
game_powers = [0,1,2,3,4,5,6,7,8,9,10]

print(f"\033[91mStarting a Grid search for hyperparameter tuning\033[0m\nTotal number of iterations: {len(maps)*len(epsilons)*len(alphas)*len(lose_costs)*len(win_costs)*len(move_costs)*len(eat_costs)*len(training_powers)}\nEach training runs {len(game_powers)} games\n")

def run_experiment(map_filename, epsilon, alpha, lose_cost, win_cost, move_cost, eat_cost, training_power, game_powers):
    """
    Function to run training & evaluation for one combination of hyperparameters.
    This is the function we will parallelize.
    """
    with open("./parallel_jobs/core_" + str(os.getpid()) + ".log", "a") as f:
        f.write("\n\n" + "#"*170 + "\n")
        f.write(f"Running training with map: {map_filename}, epsilon: {epsilon}, alpha: {alpha}, lose_cost: {lose_cost}, win_cost: {win_cost}, move_cost: {move_cost}, eat_cost: {eat_cost}, training_power: {training_power}\n")
        f.write("#"*170 + "\n\n")

    enumerator = States_enumerator(map_filename=map_filename, logging=False)
    value_iterator = Value_iterator(enumerator, alpha=alpha, epsilon=epsilon, lose_cost=lose_cost, win_cost=win_cost, move_cost=move_cost, eat_cost=eat_cost, power=training_power, control_accuracy=True, logging=False)
    value_iterator.run()
    for game_power in game_powers:
        with open("./parallel_jobs/core_" + str(os.getpid()) + ".log", "a") as f:
            f.write(f"\nRunning game with game power: {game_power}\n")
        pacmam_game = Game(value_iterator, pretrained=False, tile_size=50, fps=1000, power=game_power, measure_performance=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(os.getpid()))

if __name__ == "__main__":
    param_grid = []
    for map_filename in maps:
        for epsilon in epsilons:
            for alpha in alphas:
                for lose_cost in lose_costs:
                    for win_cost in win_costs:
                        for move_cost in move_costs:
                            for eat_cost in eat_costs:
                                for training_power in training_powers:
                                    param_grid.append((map_filename, epsilon, alpha, lose_cost, win_cost, move_cost, eat_cost, training_power, game_powers))
    Parallel(n_jobs=-1)(delayed(run_experiment)(*params) for params in param_grid)
    print("\nAll experiments finished!")
