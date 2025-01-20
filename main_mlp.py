from mlp_reinforcement_learning import Neural_Policy_iterator, Renderer, Game
from helper_functions import State_initializer

state_initializer = State_initializer(map_filename="map_2", logging=True, load_from_txt=False)
#Renderer(state_initializer, fps=15)
policy_iterator = Neural_Policy_iterator(state_initializer, random_spawn=False, max_episodes=7000, power=5, renderer=None, logging=True, increasing_power=False, alpha = 0.001, gamma = 0.95, epsilon = 1.0, min_epsilon=0.05, eat_reward = 20, move_reward=-5, lose_reward=-500, win_reward=100)
policy_iterator.load_Q()
#policy_iterator.run()
#policy_iterator.store_Q()
pacman_game = Game(policy_iterator, pretrained=True, power=5, logging=False)
pacman_game.run()