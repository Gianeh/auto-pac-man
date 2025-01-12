# Some helper methods to check states, stage costs, norm of the difference


# Check if a state is terminal, with no regard to impossibility of the state
def is_terminal(state, number_of_movables):
    return sum(state[number_of_movables:]) == 0 or state[0] in state[1:number_of_movables]

# Norm to infinity of the difference between two dictionaries
def diff_norm(dict_1, dict_2):
    if len(dict_1) != len(dict_2):
        raise ValueError("Dictionaries must have the same length")
    return max([abs(dict_1[key] - dict_2[key]) for key in dict_1])

# Evaluate if a pacman move is valid and return a new state
def pacman_move(state, action, number_of_movables, candies_positions, map):
    eaten=False
    # current position
    pacman_position = state[0]
    # next position
    new_position = (pacman_position[0] + action[0], pacman_position[1] + action[1])
    # if invalid return False
    if map[new_position[1]][new_position[0]] == 1:
        return False, eaten
    else:
        new_state = [new_position] + state[1:]
        # candy eaten check
        for candy_index in range(number_of_movables, len(state)):
            if new_position == candies_positions[candy_index] and state[candy_index] == 1:
                new_state[candy_index] = 0
                eaten=True

        return new_state, eaten
    
# Evaluate ghost state and return a list of valid actions
def ghost_move(state, ghost_index, moves, map):
    ghost_position = state[ghost_index]
    actions = [moves[a] for a in moves]
    possible_actions = []
    # exclude stay action for ghosts
    for a in actions[:-1]:
        new_position = (ghost_position[0] + a[0], ghost_position[1] + a[1])
        if map[new_position[1]][new_position[0]] == 1:  # if it is a wall
            continue
        possible_actions.append(a)
    return possible_actions

# check if a state is a win terminal state
def is_win_terminal(state, number_of_movables):
    return sum(state[number_of_movables:]) == 0 and state[0] not in state[1:number_of_movables]

# check if a state is a lose terminal state
def is_lose_terminal(state, number_of_movables):
    return state[0] in state[1:number_of_movables]

# Compute the Manhattan distance between two points
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def build_pmf(distances, power):
    # Calculate the inverse of each number ensuring that the distance is never 0 (to avoid division by 0 a +1 is added to the denominator) 
    inverse_distances = [1 / (distance**power + 1) for distance in distances]
    
    # Normalize the inverses to sum to 1
    total = sum(inverse_distances)
    pmf = [inv_dist / total for inv_dist in inverse_distances]
    
    return pmf

# Ghost move using the Manhattan distance
def ghost_move_manhattan(state, ghost_index, moves, map, power=1):
    ghost_position = state[ghost_index]
    actions = [moves[a] for a in moves]
    possible_actions = []
    manhattan_distances = []
    # exclude stay action for ghosts
    for a in actions[:-1]:
        new_position = (ghost_position[0] + a[0], ghost_position[1] + a[1])
        if map[new_position[1]][new_position[0]] == 1:  # if it is a wall
            continue
        possible_actions.append(a)
        manhattan_distances.append(manhattan_distance(new_position, state[0]))

    # Compute the pmf of actions based on the Manhattan distances wheighing more the closer actions
    pmf = build_pmf(manhattan_distances, power)
    return possible_actions, pmf



    




    