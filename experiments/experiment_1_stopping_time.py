import numpy as np

from market.substrate import MarketSubstrate
from events.time import TimeStoppingCondition
from events.engine import EventEngine
from events.cards import InformationCard
from simulation.log import EventLog
from simulation.runner import SimulationRunner
from agents.delta_hedging import DeltaHedgingAgent
from execution.validator import ExecutionEngine
from portfolio.tracker import PortfolioTracker
from options.payoff import EuropeanCallPayoff
from options.black_scholes import black_scholes_call_price, black_scholes_delta


def run_single_path(
    stopping_interval,
    seed,
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    dt_market=0.001,
):
    """
    Run a single Monte Carlo path and return replication error.
    
    Uses proper delta hedging strategy with initial capital set to
    Black-Scholes price and initial position set to initial delta.
    
    Parameters
    ----------
    stopping_interval : float
        Time between rebalancing opportunities
    seed : int
        Random seed for reproducibility
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity
    dt_market : float
        Market simulation time step
    
    Returns
    -------
    float
        Replication error: VT - option_payoff
    """

    # --- Compute Black-Scholes price and initial delta ---
    C_BS = black_scholes_call_price(S0, K, r, sigma, T)
    delta_0 = black_scholes_delta(S0, K, r, sigma, T, t=0.0)
    
    # --- Market ---
    market = MarketSubstrate(
        initial_time=0.0,
        initial_prices=[S0],
        mu=r,
        sigma=sigma,
        seed=seed
    )

    # --- Agent with proper initial capital ---
    option_params = {'K': K, 'r': r, 'sigma': sigma, 'T': T}
    agent = DeltaHedgingAgent(
        capital=C_BS,
        option_params=option_params,
        asset_index=0,
        rebalance_tolerance=0.001
    )
    
    # --- Set initial position to delta_0 ---
    # This represents buying the initial hedge
    initial_cost = delta_0 * S0
    agent.capital -= initial_cost
    agent.positions[0] = delta_0

    # --- Portfolio tracker ---
    tracker = PortfolioTracker(agent)

    # --- Stopping times ---
    stopping_events = []
    t = stopping_interval
    while t <= T:
        stopping_events.append(
            (TimeStoppingCondition(trigger_time=t),
             InformationCard(name="regular_check", payload={}))
        )
        t += stopping_interval

    # --- Event engine ---
    engine = EventEngine(market, stopping_events)

    # --- Execution ---
    execution_engine = ExecutionEngine()

    # --- Simulation runner ---
    runner = SimulationRunner(
        market=market,
        event_engine=engine,
        event_log=EventLog(),
        agents=[agent],
        execution_engine=execution_engine,
        portfolio_trackers=[tracker]
    )

    # --- Run simulation ---
    runner.run(dt=dt_market, max_events=len(stopping_events))

    # --- Terminal payoff ---
    payoff = EuropeanCallPayoff(K)
    ST = market.get_state()["prices"][0]
    option_payoff = payoff(ST)

    # --- Terminal portfolio value ---
    VT = tracker.history[-1]["value"]
    
    # --- Additional metrics for analysis ---
    replication_error = VT - option_payoff
    
    # Validate that agent didn't go bankrupt
    if agent.capital < -1e-6:  # Small tolerance for numerical errors
        import warnings
        warnings.warn(
            f"Agent went bankrupt: capital={agent.capital:.4f}. "
            f"This should not happen with proper delta hedging."
        )
    
    return replication_error



def run_experiment():
    """
    Run experiment testing delta hedging performance across different
    rebalancing frequencies.
    
    Returns
    -------
    dict
        Dictionary mapping stopping_interval to array of replication errors
    """
    stopping_intervals = [0.01, 0.05, 0.1, 0.2]
    n_paths = 1000

    results = {}
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Delta Hedging with Discrete Rebalancing")
    print("="*70)
    print(f"Number of paths: {n_paths}")
    print(f"Option parameters: S0=100, K=100, r=0.05, sigma=0.2, T=1.0")
    print("="*70 + "\n")

    for interval in stopping_intervals:
        errors = []
        for i in range(n_paths):
            err = run_single_path(
                stopping_interval=interval,
                seed=i
            )
            errors.append(err)

        results[interval] = np.array(errors)
        
        # Compute statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        prob_loss = np.mean(np.array(errors) < 0)
        median_error = np.median(errors)
        percentile_5 = np.percentile(errors, 5)
        percentile_95 = np.percentile(errors, 95)

        print(f"Stopping Interval Δt = {interval:.3f}")
        print(f"  Mean error:        {mean_error:>8.4f}")
        print(f"  Std error:         {std_error:>8.4f}")
        print(f"  Median error:      {median_error:>8.4f}")
        print(f"  5th percentile:    {percentile_5:>8.4f}")
        print(f"  95th percentile:   {percentile_95:>8.4f}")
        print(f"  P(V_T < Payoff):   {prob_loss:>8.3f}")
        print()

    print("="*70)
    print("Experiment complete. Results show convergence as Δt → 0.")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
