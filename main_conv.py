from conv_RL import Neural_Policy_iterator, State_initializer, Renderer, Game
import argparse

def train_schedule(policy_iterator, max_episodes):
    policy_iterator.max_episodes = max_episodes / 4
    policy_iterator.run()
    policy_iterator.store_Q()
    policy_iterator.episodes = 0
    policy_iterator.random_spawn=True
    policy_iterator.epsilon = 1.0
    policy_iterator.run()
    policy_iterator.store_Q()
    policy_iterator.episodes = 0
    policy_iterator.random_spawn=False
    policy_iterator.epsilon = 0.5
    policy_iterator.run()
    policy_iterator.store_Q()
    policy_iterator.episodes = 0
    policy_iterator.random_spawn=True
    policy_iterator.increasing_power = True
    policy_iterator.epsilon = policy_iterator.min_epsilon
    policy_iterator.run()   
    policy_iterator.store_Q()

def main():
   
    # parse arguments
    parser = argparse.ArgumentParser(description='Train a neural policy')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Number of episodes to train')
    args = parser.parse_args()

    #Renderer(state_initializer, fps=10)
    state_initializer = State_initializer(map_filename="ClassicGrid", load_from_txt=True, logging=True)
    policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=6000, renderer=None, logging=True, power=2, alpha = 0.0002, gamma = 0.95, epsilon = 1.0, min_epsilon=0.1, eat_reward = 10, move_reward=-1, lose_reward=-500, win_reward=100)
    policy_iterator.run()
    policy_iterator.store_Q()
    #train_schedule(policy_iterator, args.max_episodes)
    pacman_game = Game(policy_iterator, pretrained=True, logging=False, power=2)
    pacman_game.run()

if __name__ == "__main__":
    
    main()


