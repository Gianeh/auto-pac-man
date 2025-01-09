from reinforcement_learning import State_initializer, Policy_iterator, Renderer

state_initializer = State_initializer(map_filename="ez_map", logging=True)
policy_iterator = Policy_iterator(state_initializer, max_episodes=100000, renderer=Renderer(state_initializer, fps=1000), logging=True)
policy_iterator.run()