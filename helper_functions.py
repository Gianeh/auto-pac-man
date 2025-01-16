# Some helper methods to check states, stage costs, norm of the difference
import heapq

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

def a_star_distance(game_map, start, goal):
    if start == goal:
        return 0

    # Manhattan distance
    def heuristic(pos1, pos2):
        (x1, y1), (x2, y2) = pos1, pos2
        return abs(x1 - x2) + abs(y1 - y2)

    visited = set()
    # Priority queue stores (f, g, (x, y)), where f = g + h
    priority_queue = []
    start_h = heuristic(start, goal)
    heapq.heappush(priority_queue, (start_h, 0, start))  # f, g, position

    while priority_queue:
        # this returns the tuple with the smallest f
        f, g, current = heapq.heappop(priority_queue)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return g  # g is cost-so-far, i.e. actual distance

        current_x, current_y = current
        for new_x, new_y in [(current_x+1, current_y), (current_x-1, current_y), (current_x, current_y+1), (current_x, current_y-1)]:
            # Check boundaries and walls
            if 0 < new_y < len(game_map)-1 and 0 < new_x < len(game_map[0])-1:
                if game_map[new_y][new_x] != 1 and (new_x, new_y) not in visited:
                    new_g = g + 1
                    new_h = heuristic((new_x, new_y), goal)
                    new_f = new_g + new_h
                    heapq.heappush(priority_queue, (new_f, new_g, (new_x, new_y)))

    # Goal not reachable
    return 9999

def ghost_move_pathfinding(state, ghost_index, moves, game_map, power=1):
    ghost_position = state[ghost_index]

    # Collect valid actions (exclude stay action for ghosts, if you wish)
    possible_actions = []
    a_star_distances = []
    # Exclude the last item if it's 'stay'
    for action_key in list(moves.keys())[:-1]:
        dx, dy = moves[action_key]
        new_position = (ghost_position[0] + dx, ghost_position[1] + dy)

        # Skip if this new cell is a wall
        if game_map[new_position[1]][new_position[0]] == 1:
            continue

        possible_actions.append((dx, dy))

        # BFS distance from new_position to pac-man
        dist = a_star_distance(game_map, new_position, state[0])
        a_star_distances.append(dist)

    # Build pmf just like with manhattan_distance
    # e.g., if you have: build_pmf(distances, power)
    pmf = build_pmf(a_star_distances, power)

    return possible_actions, pmf



    




    