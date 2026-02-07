import numpy as np
import matplotlib.pyplot as plt

from experiments.experiment_1_stopping_time import run_experiment


def plot_histograms(results):
    """
    Plot replication error distributions for each stopping interval.
    """
    plt.figure(figsize=(10, 6))

    for interval, errors in results.items():
        plt.hist(
            errors,
            bins=50,
            density=True,
            alpha=0.4,
            label=f"Δt = {interval}"
        )

    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Replication Error  (V_T - Payoff)")
    plt.ylabel("Density")
    plt.title("Replication Error Distribution vs Stopping-Time Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_variance_scaling(results):
    """
    Plot variance of replication error vs stopping interval.
    """
    intervals = sorted(results.keys())
    variances = [np.var(results[i]) for i in intervals]

    plt.figure(figsize=(8, 5))
    plt.plot(intervals, variances, marker="o")
    plt.xlabel("Stopping Interval Δt")
    plt.ylabel("Var(Replication Error)")
    plt.title("Replication Error Variance vs Decision Frequency")
    plt.tight_layout()
    plt.show()


def plot_downside_risk(results):
    """
    Plot probability of negative replication error vs stopping interval.
    """
    intervals = sorted(results.keys())
    probs = [np.mean(results[i] < 0.0) for i in intervals]

    plt.figure(figsize=(8, 5))
    plt.plot(intervals, probs, marker="o")
    plt.xlabel("Stopping Interval Δt")
    plt.ylabel("P(Replication Error < 0)")
    plt.title("Downside Risk vs Decision Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()

    plot_histograms(results)
    plot_variance_scaling(results)
    plot_downside_risk(results)
