from conv_RL import Neural_Policy_iterator
from reinforcement_learning import Renderer, State_initializer

#Renderer(state_initializer, fps=10)

state_initializer = State_initializer(map_filename="ClassicGrid", load_from_txt=True, logging=True)
policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=4000, renderer=None, logging=True, power=1, alpha = 0.01, gamma = 0.90, epsilon = 1.0, eat_reward = 100, move_reward=-5, lose_reward=-100, win_reward=1000)

#policy_iterator.load_Q()
policy_iterator.run()
policy_iterator.store_Q()
