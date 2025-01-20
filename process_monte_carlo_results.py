# Merge all the results from the Monte Carlo simulations into a game_power files
# then create 3D histograms and barplots to visualize the results.
import os

game_powers = set()
thresholds = ["under", "between", "over"]
for filename in os.listdir("./monte_carlo_mlp"):
    if filename.endswith(".log"):
        continue
    game_powers.add(int(filename.split("_")[0]))

game_powers = sorted(list(game_powers), reverse = True)
print(game_powers)

efficiencies = {game_power: [] for game_power in game_powers}
stage_costs = {game_power: [] for game_power in game_powers}
for game_power in game_powers:
    for threshold in thresholds:
        if not os.path.exists(f"./monte_carlo_mlp/{str(game_power)}_{threshold}_threshold.txt"):
            continue
        with open(f"./monte_carlo_mlp/{str(game_power)}_{threshold}_threshold.txt", "r") as f:
            for line in f.readlines():
                efficiency = float(line.split(",")[0].split("= ")[1])
                stage_cost = float(line.split(",")[1].split("= ")[1])
                efficiencies[game_power].append(efficiency)
                stage_costs[game_power].append(stage_cost)

import matplotlib.pyplot as plt
import numpy as np
mean_costs = {game_power: sum(stage_costs[game_power]) / len(stage_costs[game_power]) for game_power in game_powers}
mean_efficiencies = {game_power: sum(efficiencies[game_power]) / len(efficiencies[game_power]) for game_power in game_powers}

# Bar plot of mean costs by power
fig = plt.figure()
plt.bar(game_powers, [mean_costs[game_power] for game_power in mean_costs])
plt.show()


all_stage_costs = []
for gp in game_powers:
    all_stage_costs.extend(stage_costs[gp])

num_bins = 50
bins = np.linspace(min(all_stage_costs), max(all_stage_costs), num_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

bar_width = (bins[1] - bins[0]) * 0.9
colors = plt.cm.tab10(np.linspace(0, 1, len(game_powers)))

for y_index, gp in enumerate(game_powers):
    counts, _ = np.histogram(stage_costs[gp], bins=bins)
    color = colors[y_index % len(colors)]
    for i in range(num_bins):
        x = bin_centers[i]
        y = y_index
        z = 0
        dx = bar_width
        dy = 0.6
        dz = counts[i]
        ax.bar3d(x, y, z, dx, dy, dz, color=color, alpha=0.7)

ax.set_xlabel('Cumulative Reward')
ax.set_ylabel('Game Power')
ax.set_zlabel('Frequency')

tick_positions = bin_centers[::3]  # Only show some bins
ax.set_xticks(tick_positions)
ax.set_xticklabels([f"{val:.1e}" for val in tick_positions])

ax.set_yticks(range(len(game_powers)))
ax.set_yticklabels(game_powers)

plt.title("3D Distribution")
plt.tight_layout()
plt.show()