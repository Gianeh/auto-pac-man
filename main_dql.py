from deep_reinforcement_learning import State_initializer, Neural_Policy_iterator
from reinforcement_learning import Renderer

state_initializer = State_initializer(map_filename="ez_map", logging=True)
policy_iterator = Neural_Policy_iterator(state_initializer, max_episodes=5000000, renderer=Renderer(state_initializer, fps=15), logging=True, power=2)
policy_iterator.run()
#policy_iterator.store_Q()