import numpy as np
import matplotlib.pyplot as plt

from experiments.experiment_1_stopping_time import run_single_path
from options.black_scholes import black_scholes_call_price


def probability_of_success(
    initial_capital,
    stopping_interval,
    n_paths=1000,
):
    successes = 0

    for i in range(n_paths):
        error = run_single_path(
            initial_capital=initial_capital,
            stopping_interval=stopping_interval,
            seed=i,
        )

        if error >= 0:
            successes += 1

    return successes / n_paths

def run_market_comparison():
    # --- Market / option inputs (replace with real data later) ---
    S0 = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.20

    # Observed market price (example placeholder)
    market_price = 8.20

    # --- Black–Scholes benchmark ---
    C_BS = black_scholes_call_price(S0, K, r, sigma, T)

    print(f"Black–Scholes price: {C_BS:.2f}")
    print(f"Market price:        {market_price:.2f}")

    # --- Sweep initial capital ---
    stopping_interval = 0.05
    capital_grid = np.linspace(C_BS, C_BS + 3.0, 13)

    success_probs = []

    for c in capital_grid:
        p = probability_of_success(
            initial_capital=c,
            stopping_interval=stopping_interval,
            n_paths=1000,
        )
        success_probs.append(p)
        print(f"Capital={c:.2f}, P(success)={p:.3f}")

    # --- Extract realizable cost ---
    target_prob = 0.95
    realizable_cost = None

    for c, p in zip(capital_grid, success_probs):
        if p >= target_prob:
            realizable_cost = c
            break

    print(f"\nRealizable replication cost (95%): {realizable_cost:.2f}")

    # --- Plot ---
    plt.plot(capital_grid, success_probs, marker="o")
    plt.axhline(target_prob, linestyle="--", label="95% success")
    plt.axvline(C_BS, linestyle="--", label="BS price")
    plt.axvline(market_price, linestyle="--", label="Market price")
    plt.axvline(realizable_cost, linestyle="--", label="Realizable cost")

    plt.xlabel("Initial Capital")
    plt.ylabel("P(V_T ≥ Payoff)")
    plt.title("Realizable Hedging Cost vs BS and Market Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_market_comparison()


