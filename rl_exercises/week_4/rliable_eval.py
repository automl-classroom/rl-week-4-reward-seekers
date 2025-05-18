import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import rliable.library as rlib
import rliable.metrics as rmetrics
from hydra.utils import get_original_cwd
from rl_exercises.week_4.dqn import DQNAgent, set_seed

# Configuration
SEEDS = [0, 1, 2, 3, 4]
NUM_FRAMES = 20000
EVAL_INTERVAL = 1000
ENV_NAME = "CartPole-v1"
RESULTS_DIR = os.path.join(get_original_cwd(), "rliable_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(seed: int) -> list:
    env = gym.make(ENV_NAME)
    set_seed(env, seed)
    agent = DQNAgent(env=env, seed=seed)
    _, rewards = agent.train(num_frames=NUM_FRAMES, eval_interval=EVAL_INTERVAL)
    return rewards


def aggregate_runs() -> dict:
    all_runs = {}
    for seed in SEEDS:
        print(f"Running seed {seed}")
        rewards = run_experiment(seed)
        all_runs[f"seed_{seed}"] = np.array(rewards)
    return all_runs


def plot_rliable_metrics(scores: dict):
    # Align data by padding with NaNs
    max_len = max(len(v) for v in scores.values())
    for k in scores:
        scores[k] = np.pad(
            scores[k], (0, max_len - len(scores[k])), constant_values=np.nan
        )

    score_matrix = np.array(list(scores.values()))
    score_matrix = np.nan_to_num(score_matrix, nan=0.0)  # RLiable requires no NaNs

    score_dict = {"DQN": score_matrix}

    # Metrics: IQM, Mean, Median, Optimality Gap
    aggregate_func = rmetrics.AggregateMetrics(
        rmetrics.aggregate_iqm,
        rmetrics.aggregate_mean,
        rmetrics.aggregate_median,
        rmetrics.aggregate_optimality_gap,
    )
    aggregate_scores, score_cis = rlib.compute_interval_estimates(
        score_dict, aggregate_func, reps=500
    )

    # Plotting
    rmetrics.plot_score_distribution(score_dict, ["DQN"])
    plt.title("Score Distribution Across Seeds")
    plt.savefig(os.path.join(RESULTS_DIR, "score_distribution.png"))

    rmetrics.plot_performance_profiles(score_dict, ["DQN"])
    plt.title("Performance Profile")
    plt.savefig(os.path.join(RESULTS_DIR, "performance_profile.png"))

    print("Aggregate Scores (DQN):")
    for metric in aggregate_scores["DQN"]:
        print(
            f"{metric}: {aggregate_scores['DQN'][metric]:.2f} ± {score_cis['DQN'][metric]:.2f}"
        )


if __name__ == "__main__":
    scores = aggregate_runs()
    plot_rliable_metrics(scores)
