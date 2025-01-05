from dynamic_programming import States_enumerator, Value_iterator, Game


enumerator = States_enumerator(map_filename="ez_map", logging=True)
# Good enough parameters
#value_iterator = Value_iterator(enumerator, alpha=0.9, delta=0.01, epsilon=0.1, lose_cost=10000, win_cost=-10000, move_cost=10, eat_cost=-1e1, power=0, control_accuracy=True, logging=True, stay_penalty=0)
# Very good parameters
#value_iterator = Value_iterator(enumerator, alpha=0.9, delta=0.01, epsilon=0.1, lose_cost=20000, win_cost=-10000, move_cost=10, eat_cost=-1e1, power=0, control_accuracy=True, logging=True, stay_penalty=0)
value_iterator = Value_iterator(enumerator, alpha=0.9, delta=0.01, epsilon=0.1, lose_cost=20000, win_cost=-10000, move_cost=10, eat_cost=-1e1, power=10, control_accuracy=True, logging=True, stay_penalty=0)
#value_iterator.run()
#value_iterator.store_policy()
#value_iterator.store_value_function()
pacmam_game = Game(value_iterator, pretrained=True, tile_size=50, fps=10, logging=True, power=10)
pacmam_game.run(ghost_controlled=True, loop_till_loss=True)