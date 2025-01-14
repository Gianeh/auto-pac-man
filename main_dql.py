from deep_reinforcement_learning import Neural_Policy_iterator
from reinforcement_learning import Renderer, State_initializer

state_initializer = State_initializer(map_filename="ez_map", logging=True)
#Renderer(state_initializer, fps=15)
policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=100000, renderer=Renderer(state_initializer, fps=15), logging=True, power=10, alpha = 0.001, gamma = 0.8, epsilon = 0.0, eat_reward = 1000, move_reward=-10, lose_reward=-500, win_reward=10000)
policy_iterator.load_Q()
policy_iterator.run()
policy_iterator.store_Q()