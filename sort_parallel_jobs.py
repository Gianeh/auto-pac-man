import os
import re

# A script to sort the parallel log files from the folder parallel_jobs/ into a single file.

def parse_efficiency(line: str) -> float:
    """
    Extracts the numeric value following 'efficiency =' from a line.
    Returns 0.0 if not found.
    Example line:
    ez_map - efficiency = 0.05464895635673624, number_of_moves = 5270, ...
    """
    match = re.search(r'efficiency\s*=\s*([\d\.]+)', line)
    if match:
        return float(match.group(1))
    return 0.0

# Prepare the results directory
os.makedirs("./results", exist_ok=True)

# For collecting the global lines (across all PIDs)
global_under = []
global_between = []
global_over = []

# first pass: gather unique pids:
all_pids = set()
for filename in os.listdir("./parallel_jobs"):
    if filename.endswith(".txt"):
        pid = filename.split("_")[0]
        all_pids.add(pid)

# For each PID, gather under, between, over lines and write them out.
for pid in all_pids:
    under_file = f"./parallel_jobs/{pid}_under_threshold.txt"
    between_file = f"./parallel_jobs/{pid}_between_threshold.txt"
    over_file = f"./parallel_jobs/{pid}_over_threshold.txt"

    # Check if they exist before opening
    under, between, over = [], [], []

    # Read under
    if os.path.isfile(under_file):
        with open(under_file, "r") as f:
            under = f.readlines()

    # Read between
    if os.path.isfile(between_file):
        with open(between_file, "r") as f:
            between = f.readlines()

    # Read over
    if os.path.isfile(over_file):
        with open(over_file, "r") as f:
            over = f.readlines()

    # (Optionally) remove the original logs if you no longer need them
    # if os.path.isfile(under_file):
    #     os.remove(under_file)
    # if os.path.isfile(between_file):
    #     os.remove(between_file)
    # if os.path.isfile(over_file):
    #     os.remove(over_file)

    # Step 2: Write the merged logs for each PID to ./results/{pid}.log
    merged_file = f"./results/{pid}.log"
    with open(merged_file, "w") as f:
        f.write(f"\n\n{'#' * 170}\n")
        f.write(f"PID: {pid}\n")
        f.write("#" * 170 + "\n\n")
        f.write(f"{len(under)} under threshold\n")
        f.write("".join(under))   # or f.write("\n".join(under))
        f.write(f"{len(between)} between threshold\n")
        f.write("".join(between))
        f.write(f"{len(over)} over threshold\n")
        f.write("".join(over))
    print(f"Finished sorting PID {pid}")

    # Step 3: Collect them into global lists for final sorting
    # We store (efficiency, line) pairs to make sorting easier.
    for line in under:
        global_under.append((parse_efficiency(line), line))
    for line in between:
        global_between.append((parse_efficiency(line), line))
    for line in over:
        global_over.append((parse_efficiency(line), line))

# Step 4: Sort each global list by efficiency
global_under.sort(key=lambda x: x[0], reverse=True)   # descending by efficiency
global_between.sort(key=lambda x: x[0], reverse=True)
global_over.sort(key=lambda x: x[0], reverse=True)

# Step 5: Write them all into a single final file, e.g. "./results/final_results.log"
final_file = "./results/final_results.log"
with open(final_file, "w") as f:
    f.write("#" * 80 + "\n")
    f.write("UNDER THRESHOLD (sorted by efficiency)\n")
    f.write("#" * 80 + "\n\n")
    for eff, line in global_under:
        f.write(line)

    f.write("\n\n" + "#" * 80 + "\n")
    f.write("BETWEEN THRESHOLD (sorted by efficiency)\n")
    f.write("#" * 80 + "\n\n")
    for eff, line in global_between:
        f.write(line)

    f.write("\n\n" + "#" * 80 + "\n")
    f.write("OVER THRESHOLD (sorted by efficiency)\n")
    f.write("#" * 80 + "\n\n")
    for eff, line in global_over:
        f.write(line)

print(f"Created final sorted file: {final_file}")

# Step 6: Among all the global_over lines, find the n parameter sets with the highest efficiency sum across all game powers.
n = 10
best_trainings = []
parameters = set()
for _, line in global_over:
    parameter = "alpha"
    for p in line.split("alpha")[1].split(", game_power")[0]:
        parameter += p
    parameters.add(parameter)  # remove the game power value

print(len(parameters), "unique parameter sets found.")


# Not all the parameter sets that scored at least an over threshold game did score the same for all games.
global_trainings = global_over + global_between + global_under

for unique_parameter in parameters:
    efficiency_sum = 0
    number_of_games = 0
    for efficiency, line in global_trainings:
        if unique_parameter in line:
            efficiency_sum += efficiency
            number_of_games += 1
    best_trainings.append((efficiency_sum / number_of_games, unique_parameter))
    #print(f"Parameter set: {unique_parameter} - Efficiency sum: {efficiency_sum} - Number of games: {number_of_games}")

# Sort the best_trainings by efficiency mean and pick n best
best_trainings.sort(key=lambda x: x[0], reverse=True)
best_trainings = best_trainings[:n]

# Write the top n to a file
top_n_file = f"./results/top_{n}_trainings.log"
with open(top_n_file, "w") as f:
    f.write("#" * 80 + "\n")
    f.write(f"TOP {n} TRAININGS (sorted by efficiency mean against all game powers tested)\n")
    f.write("#" * 80 + "\n\n")
    for eff_mean, line in best_trainings:
        f.write(f"Efficiency mean: {eff_mean} - {line}\n")


# Step 7: Create histograms of the efficiency values for each game power of the best n trainings
from matplotlib import pyplot as plt

# Note: the number_of_games is exactly the number of tested game powers

histograms = []
for eff_mean, param in best_trainings:
    game_efficiencies = []
    game_powers = []
    for eff, training in global_trainings:
        if param in training:
            game_power = int(training.split("game_power = ")[1])
            game_powers.append((eff, game_power))

    game_powers.sort(key=lambda x: x[1])    # ascending by game power
    for eff, game_power in game_powers:
        game_efficiencies.append(eff)

    histograms.append((param, game_powers, game_efficiencies, eff_mean))

# Plot the histograms
for param, game_powers, game_efficiencies, eff_mean in histograms:
    fig = plt.figure()
    title = ",".join(param.split(",")[:3]) + ",\n" + ",".join(param.split(",")[3:])
    plt.bar([str(game_power) for _, game_power in game_powers], game_efficiencies)
    # add the mean efficiency level to the plot
    plt.axhline(y=eff_mean, color='r', linestyle='-', label=f"Mean efficiency: {eff_mean}")
    plt.legend(loc='lower left')
    plt.xlabel("Game power")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency distribution for:\n\n{title}\n")
    # save the figure
    plt.savefig(f"./results/{histograms.index((param, game_powers, game_efficiencies, eff_mean)) + 1}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Finished plotting histogram for {param}")

print("Finished plotting histograms.")
    
