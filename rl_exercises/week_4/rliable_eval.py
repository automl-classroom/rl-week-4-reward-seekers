"""import itertools
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rliable.metrics as metrics
import rliable.plotting as rly
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import library as rly_lib

# Set experiment parameters
env_name = "CartPole-v1"
seeds = [0, 1, 2, 3, 4]
depths = [1]
widths = [64]
buffer_sizes = [10000]
batch_sizes = [32]
num_frames = 20000
eval_interval = 1000

plots_dir = "plots_eval"
os.makedirs(plots_dir, exist_ok=True)

combinations = list(itertools.product(depths, widths, buffer_sizes, batch_sizes))
results_dict = {}

for depth, width, buffer_capacity, batch_size in combinations:
    algo_name = f"D{depth}_W{width}_B{batch_size}_Buf{buffer_capacity}"
    print(f"\nEvaluating config: {algo_name}")
    seed_rewards = []

    for seed in seeds:
        env = gym.make(env_name)
        set_seed(env, seed)

        agent = DQNAgent(
            env=env,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_final=0.01,
            epsilon_decay=500,
            target_update_freq=1000,
            seed=seed,
            hidden_dim=width,
            depth=depth,
        )

        _, y_mean_rewards = agent.train(
            num_frames=num_frames, eval_interval=eval_interval
        )

        # Make all reward curves the same length
        max_len = num_frames // eval_interval
        padded_rewards = y_mean_rewards + [y_mean_rewards[-1]] * (
            max_len - len(y_mean_rewards)
        )
        seed_rewards.append(padded_rewards)

    results_dict[algo_name] = np.array(seed_rewards)

# Use RLiable to compute metrics
aggregate_fn = lambda scores: metrics.aggregate_metrics(
    scores,
    metrics=[metrics.mean, metrics.median, metrics.iqm, metrics.optimality_gap],
    normalize_by=500.0,  # max score for CartPole-v1
)

agg_scores, score_cis = rly_lib.get_interval_estimates(results_dict, aggregate_fn)

# Print summary table
print("\n=== Evaluation Summary ===")
for algo, values in agg_scores.items():
    print(f"\nAlgorithm: {algo}")
    for metric, val in values.items():
        ci = score_cis[algo][metric]
        print(f"{metric}: {val:.2f} ± {ci:.2f}")

# Plot score distribution
rly.plot_score_distribution(results_dict, metric="IQM", normalize_by=500.0)
plt.title("Score Distribution (IQM)")
plt.savefig(os.path.join(plots_dir, "score_distribution.png"))
plt.close()

# Plot performance profile
rly.plot_performance_profiles(results_dict, normalize_by=500.0)
plt.title("Performance Profile")
plt.savefig(os.path.join(plots_dir, "performance_profile.png"))
plt.close()
"""
