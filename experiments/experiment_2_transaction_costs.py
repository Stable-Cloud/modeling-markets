# experiments/experiment_2_transaction_costs.py

import numpy as np

from market.substrate import MarketSubstrate
from events.time import TimeStoppingCondition
from events.engine import EventEngine
from events.cards import InformationCard
from simulation.log import EventLog
from simulation.runner import SimulationRunner
from agents.delta_hedging import DeltaHedgingAgent
from execution.validator import ExecutionEngine
from execution.costs import ProportionalCostModel
from portfolio.tracker import PortfolioTracker
from options.payoff import EuropeanCallPayoff
from options.black_scholes import black_scholes_call_price, black_scholes_delta


def run_single_path_with_costs(
    stopping_interval,
    transaction_cost_bps,
    seed,
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    dt_market=0.001,
):
    """
    Run a single Monte Carlo path with transaction costs.
    
    Parameters
    ----------
    stopping_interval : float
        Time between rebalancing opportunities
    transaction_cost_bps : float
        Transaction cost rate in basis points
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
    dict
        Dictionary with replication_error, total_costs, trade_count
    """
    
    # Compute Black-Scholes price and initial delta
    C_BS = black_scholes_call_price(S0, K, r, sigma, T)
    delta_0 = black_scholes_delta(S0, K, r, sigma, T, t=0.0)
    
    # Market
    market = MarketSubstrate(
        initial_time=0.0,
        initial_prices=[S0],
        mu=r,
        sigma=sigma,
        seed=seed
    )

    # Agent with proper initial capital
    option_params = {'K': K, 'r': r, 'sigma': sigma, 'T': T}
    agent = DeltaHedgingAgent(
        capital=C_BS,
        option_params=option_params,
        asset_index=0,
        rebalance_tolerance=0.001
    )
    
    # Set initial position to delta_0
    initial_cost = delta_0 * S0
    agent.capital -= initial_cost
    agent.positions[0] = delta_0

    # Portfolio tracker
    tracker = PortfolioTracker(agent)

    # Stopping times
    stopping_events = []
    t = stopping_interval
    while t <= T:
        stopping_events.append(
            (TimeStoppingCondition(trigger_time=t),
             InformationCard(name="regular_check", payload={}))
        )
        t += stopping_interval

    # Event engine
    engine = EventEngine(market, stopping_events)

    # Execution with transaction costs
    if transaction_cost_bps > 0:
        cost_model = ProportionalCostModel(rate_bps=transaction_cost_bps)
    else:
        cost_model = None
    
    execution_engine = ExecutionEngine(cost_model=cost_model)

    # Simulation runner
    runner = SimulationRunner(
        market=market,
        event_engine=engine,
        event_log=EventLog(),
        agents=[agent],
        execution_engine=execution_engine,
        portfolio_trackers=[tracker]
    )

    # Run simulation
    runner.run(dt=dt_market, max_events=len(stopping_events))

    # Terminal payoff
    payoff = EuropeanCallPayoff(K)
    ST = market.get_state()["prices"][0]
    option_payoff = payoff(ST)

    # Terminal portfolio value
    VT = tracker.history[-1]["value"]
    
    # Metrics
    replication_error = VT - option_payoff
    total_costs = agent.total_transaction_costs
    trade_count = agent.trade_count
    
    return {
        'replication_error': replication_error,
        'total_costs': total_costs,
        'trade_count': trade_count,
        'final_capital': agent.capital,
        'final_value': VT
    }


def run_experiment():
    """
    Run experiment testing impact of transaction costs on hedging performance.
    
    Returns
    -------
    dict
        Nested dictionary: {cost_rate: {stopping_interval: results}}
    """
    stopping_intervals = [0.05, 0.1, 0.2]
    cost_rates_bps = [0, 5, 10, 20, 50]  # 0%, 0.05%, 0.1%, 0.2%, 0.5%
    n_paths = 1000
    
    results = {}
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Transaction Costs on Delta Hedging")
    print("="*70)
    print(f"Number of paths: {n_paths}")
    print(f"Option parameters: S0=100, K=100, r=0.05, sigma=0.2, T=1.0")
    print("="*70 + "\n")
    
    for cost_bps in cost_rates_bps:
        results[cost_bps] = {}
        
        print(f"\n{'='*70}")
        print(f"Transaction Cost Rate: {cost_bps} bps ({cost_bps/100:.2f}%)")
        print(f"{'='*70}\n")
        
        for interval in stopping_intervals:
            path_results = []
            
            for i in range(n_paths):
                result = run_single_path_with_costs(
                    stopping_interval=interval,
                    transaction_cost_bps=cost_bps,
                    seed=i
                )
                path_results.append(result)
            
            # Aggregate statistics
            errors = [r['replication_error'] for r in path_results]
            costs = [r['total_costs'] for r in path_results]
            trades = [r['trade_count'] for r in path_results]
            
            results[cost_bps][interval] = {
                'errors': np.array(errors),
                'costs': np.array(costs),
                'trades': np.array(trades)
            }
            
            # Compute statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            mean_cost = np.mean(costs)
            mean_trades = np.mean(trades)
            prob_loss = np.mean(np.array(errors) < 0)
            
            # Effective cost = mean_error + mean_cost (total drag on performance)
            effective_cost = mean_error + mean_cost
            
            print(f"Stopping Interval Δt = {interval:.3f}")
            print(f"  Mean error:           {mean_error:>8.4f}")
            print(f"  Std error:            {std_error:>8.4f}")
            print(f"  Mean transaction cost:{mean_cost:>8.4f}")
            print(f"  Mean trade count:     {mean_trades:>8.1f}")
            print(f"  Effective cost:       {effective_cost:>8.4f}")
            print(f"  P(V_T < Payoff):      {prob_loss:>8.3f}")
            print()
    
    print("="*70)
    print("Key Insights:")
    print("- Transaction costs increase effective hedging cost")
    print("- Trade-off: frequent rebalancing (lower gamma error) vs.")
    print("  infrequent rebalancing (lower transaction costs)")
    print("- Optimal rebalancing frequency depends on cost rate")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
