from deep_reinforcement_learning import Neural_Policy_iterator
from reinforcement_learning import Renderer, State_initializer

state_initializer = State_initializer(map_filename="ez_map", logging=True)
#Renderer(state_initializer, fps=15)
policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=50000, renderer=Renderer(state_initializer, fps=15), logging=True, power=2, alpha = 0.001, gamma = 0.9, epsilon = 0.1, eat_reward = 500, move_reward=-1, lose_reward=-100, win_reward=10000)
policy_iterator.load_Q()
policy_iterator.run()
policy_iterator.store_Q()