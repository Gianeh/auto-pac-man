from reinforcement_learning import State_initializer, Policy_iterator, Renderer, Game

state_initializer = State_initializer(map_filename="stupid_map", logging=True, load_from_txt=True)
#policy_iterator = Policy_iterator(state_initializer, max_episodes=10000, renderer=Renderer(state_initializer, fps=1), logging=True)

policy_iterator = Policy_iterator(state_initializer, max_episodes=50000, renderer=None, logging=True, alpha = 0.3, epsilon = 0.5, power = 2)
policy_iterator.load_Q()
policy_iterator.run()
policy_iterator.store_Q()
policy_iterator.load_Q()
#print(policy_iterator.Q.items())
pacman_game = Game(policy_iterator, pretrained=True, tile_size=50, fps=10, logging=True, power=2, measure_performance=False, monte_carlo=False)
pacman_game.run(loop_till_loss=True, ghost_controlled=False)
