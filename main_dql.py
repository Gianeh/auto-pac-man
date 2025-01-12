from deep_reinforcement_learning import State_initializer, Neural_Policy_iterator
from reinforcement_learning import Renderer

state_initializer = State_initializer(map_filename="ez_map", logging=True)
policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=5000000, renderer=None, logging=True, power=2, alpha = 0.3, gamma = 0.77, epsilon = 0.9, eat_reward = 500, move_reward=-20, lose_reward=-2000, win_reward=10000)
policy_iterator.run()
policy_iterator.store_Q()