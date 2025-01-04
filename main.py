from dynamic_programming import States_enumerator, Value_iterator, Game


enumerator = States_enumerator(map_filename="map1", laod_from_txt=True, logging=True)
value_iterator = Value_iterator(enumerator, alpha=0.9, delta=0.01, epsilon=0.1, lose_cost=10000, win_cost=-1000, move_cost=10, eat_cost=-1e1, power=5, control_accuracy=True, logging=True)
#value_iterator.run()
#value_iterator.store_policy()
#value_iterator.store_value_function()
pacmam_game = Game(value_iterator, pretrained=True, tile_size=40, fps=10, logging=True)
pacmam_game.run(ghost_controlled=False, loop_till_loss=True)