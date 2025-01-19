from joblib import Parallel, delayed
from helper_functions import State_initializer
from cnn_reinforcement_learning import Neural_Policy_iterator, Game
import os
from multiprocessing import Manager

game_powers = [0, 2, 4, 6, 8, 10]
simulations_number = 100

def run_experiment(policy_iterator, simulations_number, game_power, progress_dict, progress_lock, total):
    log_dir = "./monte_carlo_mlp/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

    log_path = os.path.join(log_dir, f"game_power_{game_power}.log")
    with open(log_path, "a") as f:
        f.write("\n\n" + "#" * 170 + "\n")
        f.write(f"Running Monte Carlo simulation with map: {policy_iterator.filename} and game_power = {game_power}\n")
        f.write("#" * 170 + "\n\n")

    for simulation in range(1, simulations_number + 1):
        with open(log_path, "a") as f:
            f.write(f"\nRunning simulation: {simulation}\n")
        pacmam_game = Game(policy_iterator, pretrained=True, tile_size=50, power=game_power, measure_performance=True, monte_carlo=True)
        pacmam_game.run(ghost_controlled=False, loop_till_loss=True, measure_filename=str(game_power))

    # Increment the global progress
    with progress_lock:
        progress_dict["done"] += 1
        done = progress_dict["done"]
    print(f"[PID {os.getpid()}] Completed iteration {done} / {total}.")


if __name__ == "__main__":
    state_initializer = State_initializer(map_filename="map_1", logging=True)
    policy_iterator = Neural_Policy_iterator(state_initializer, random_spawn=False, max_episodes=6000, power=2, renderer=None, logging=True, increasing_power=False, alpha = 0.001, gamma = 0.95, epsilon = 1.0, min_epsilon=0.05, eat_reward = 20, move_reward=-5, lose_reward=-500, win_reward=100)
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
