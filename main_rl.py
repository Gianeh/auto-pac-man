from reinforcement_learning import State_initializer, Policy_iterator, Renderer, Game
import argparse

def main(filename, train, load_checkpoint, max_episodes, render_training, logging, alpha, gamma, epsilon, power, eat_reward, move_reward, lose_reward, win_reward, game_power, loop_till_loss, ghost_controlled):
    # Initialize environment
    state_initializer = State_initializer(map_filename=filename, logging=True)
    renderer = None
    if render_training:
        renderer = Renderer(state_initializer, tile_size=50, fps=15)
    policy_iterator = Policy_iterator(state_initializer, max_episodes=max_episodes,
                                        renderer=renderer,
                                        logging=logging, alpha=alpha, gamma=gamma, epsilon=epsilon, power=power,
                                        eat_reward = eat_reward, move_reward=move_reward, lose_reward=lose_reward, win_reward=win_reward)
    if train:
        if load_checkpoint: policy_iterator.load_Q()
        policy_iterator.run()
        policy_iterator.store_Q()
    pacman_game = Game(policy_iterator, pretrained=True, tile_size=50, fps=15, logging=logging, power=game_power)
    pacman_game.run(loop_till_loss=loop_till_loss, ghost_controlled=ghost_controlled)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="map_1", help="The filename of the map to load")
    parser.add_argument("--train", default=False, action='store_true', help="Whether to train the model")
    parser.add_argument("--load_checkpoint", default=False, action='store_true', help="Whether to load a checkpoint")
    parser.add_argument("--max_episodes", type=int, default=500000, help="The maximum number of episodes to run")
    parser.add_argument("--render_training", default=False, action='store_true', help="Whether to render the training")
    parser.add_argument("--logging", default=False, action='store_true', help="Whether to log the training")

    # Training hyperparameters
    parser.add_argument("--alpha", type=float, default=0.05, help="The learning rate")
    parser.add_argument("--gamma", type=float, default=0.8, help="The discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The exploration rate")
    parser.add_argument("--power", type=int, default=5, help="The power of the distance function")
    parser.add_argument("--eat_reward", type=int, default=1000, help="The reward for eating a candy")
    parser.add_argument("--move_reward", type=int, default=-10, help="The reward for moving")
    parser.add_argument("--lose_reward", type=int, default=-500, help="The reward for losing")
    parser.add_argument("--win_reward", type=int, default=10000, help="The reward for winning")

    # Game Parameters
    parser.add_argument("--game_power", type=int, help="The power of the distance function for the game (default is the --power arg chosen at run time)")
    args = parser.parse_args()
    if args.game_power is None:
        args.game_power = args.power
    parser.add_argument("--loop_till_loss", default=False, action='store_true', help="Whether to loop till loss")
    parser.add_argument("--ghost_controlled", default=False, action='store_true', help="Whether the ghost is controlled by the model")


    main(**vars(parser.parse_args()))
