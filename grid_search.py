from dynamic_programming import States_enumerator, Value_iterator, Game

# Grid search for hyperparameter tuning
maps = ["map_1"]
epsilons = [1, 0.1]
alphas = [0.7, 0.9]
lose_costs = [10000, 20000, 50000]
win_costs = [-10000, -20000, -50000]
move_costs = [1, 10, 100]
eat_costs = [-10, -100]
training_powers = [0, 10]   # 0 -> uniform ghost move distribution, 10 -> ghost move towards pacman (manhattan distance)
game_powers = [0,1,2,3,4,5,6,7,8,9,10]

print(f"\033[91mStarting a Grid search for hyperparameter tuning\033[0m\nTotal number of iterations: {len(maps)*len(epsilons)*len(alphas)*len(lose_costs)*len(win_costs)*len(move_costs)*len(eat_costs)*len(training_powers)}\nEach training runs {len(game_powers)} games\n")

# Grid search
for map in maps:
    for epsilon in epsilons:
        for alpha in alphas:
            for lose_cost in lose_costs:
                for win_cost in win_costs:
                    for move_cost in move_costs:
                        for eat_cost in eat_costs:
                            for training_power in training_powers:
                                print("\n\n"+"#"*170)
                                print(f"\033[92mRunning training with map: {map}, epsilon: {epsilon}, alpha: {alpha}, lose_cost: {lose_cost}, win_cost: {win_cost}, move_cost: {move_cost}, eat_cost: {eat_cost}, training_power: {training_power}\033[0m")
                                print("#"*170, end="\n\n")
                                enumerator = States_enumerator(map_filename=map, logging=False)
                                value_iterator = Value_iterator(enumerator, alpha=alpha, epsilon=epsilon, lose_cost=lose_cost, win_cost=win_cost, move_cost=move_cost, eat_cost=eat_cost, power=training_power, control_accuracy=True, logging=True)
                                value_iterator.run()
                                value_iterator.store_policy()
                                value_iterator.store_value_function()
                                for game_power in game_powers:
                                    print(f"\033[93mRunning game with game power: {game_power}\033[0m")
                                    pacmam_game = Game(value_iterator, pretrained=True, tile_size=50, fps=1000, power=game_power, measure_performance=True)
                                    pacmam_game.run(ghost_controlled=False, loop_till_loss=True)