from dynamic_programming import States_enumerator, Value_iterator, Game
import argparse

def main(filename, train, logging, alpha, delta, epsilon, lose_cost, win_cost, move_cost, eat_cost, power, game_power, ghost_controlled, loop_till_loss):
    enumerator = States_enumerator(map_filename=filename, logging=logging)
    control_accuracy = epsilon is not None
    value_iterator = Value_iterator(enumerator, alpha=alpha, delta=delta, epsilon=epsilon, lose_cost=lose_cost, win_cost=win_cost, move_cost=move_cost, eat_cost=eat_cost, power=power, control_accuracy=control_accuracy, logging=logging)
    if train:
        value_iterator.run()
        value_iterator.store_policy()
        value_iterator.store_value_function()
    pacman_game = Game(value_iterator, pretrained=True, tile_size=50, fps=10, logging=logging, power=game_power)
    pacman_game.run(ghost_controlled=ghost_controlled, loop_till_loss=loop_till_loss)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="map_1", help="The filename of the map to load")
    parser.add_argument("--train", default=False, action='store_true', help="Whether to train the model")
    parser.add_argument("--logging", default=False, action='store_true', help="Whether to log the training")

    # Training hyperparameters
    parser.add_argument("--alpha", type=float, default=0.9, help="The discount factor")
    parser.add_argument("--delta", type=float, default=0.01, help="The maximum value of absolute difference between two iterations of the Value function")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The accuracy of the value function from the optimum")
    parser.add_argument("--power", type=int, default=2, help="The power of the distance function")
    parser.add_argument("--eat_cost", type=int, default=-1, help="The cost for eating a candy")
    parser.add_argument("--move_cost", type=int, default=1, help="The cost for moving")
    parser.add_argument("--lose_cost", type=int, default=200, help="The cost for losing")
    parser.add_argument("--win_cost", type=int, default=-500, help="The cost for winning")


    # Game Parameters
    parser.add_argument("--loop_till_loss", default=False, action='store_true', help="Whether to loop till loss")
    parser.add_argument("--ghost_controlled", default=False, action='store_true', help="Whether the ghost is controlled by the model")
    parser.add_argument("--game_power", type=int, help="The power of the distance function for the game (default is the --power arg chosen at run time)")
    args = parser.parse_args()
    if args.game_power is None:
        args.game_power = args.power


    main(**vars(parser.parse_args()))