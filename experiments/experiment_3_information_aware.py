import numpy as np 
from market.substrate import MarketSubstrate 
from events.time import TimeStoppingCondition 
from events.engine import EventEngine 
from events.cards import InformationCard 
from events.information_generator import InformationGenerator 
from simulation.log import EventLog 
from simulation.runner import SimulationRunner 
from agents.information_aware_hedging import InformationAwareDeltaHedgingAgent 
from execution.validator import ExecutionEngine 
from portfolio.tracker import PortfolioTracker 
from options.payoff import EuropeanCallPayoff 
from options.black_scholes import black_scholes_call_price, black_scholes_delta 
def run_single_path_information_aware( 
    stopping_interval, 
    use_information=True, 
    seed=None, 
    S0=100.0, 
    K=100.0, 
    r=0.05,
    sigma=0.2, 
    T=1.0,
    dt_market=0.001, 
): 
    """ 
    Run single path with information-aware agent. 
    Parameters 
    ---------- 
    use_information : bool 
        Whether to use information-aware strategies 
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
    # Agent 
    option_params = {'K': K, 'r': r, 'sigma': sigma, 'T': T} 
    if use_information: 
        agent = InformationAwareDeltaHedgingAgent( 
            capital=C_BS, 
            option_params=option_params, 
            asset_index=0, 
            base_tolerance=0.001, 
            volatility_adaptation=True, 
            gamma_adaptation=True, 
            cost_optimization=True 
        ) 
    else: 
        # Use standard agent for comparison 
        from agents.delta_hedging import DeltaHedgingAgent 
        agent = DeltaHedgingAgent( 
            capital=C_BS, 
            option_params=option_params, 
            asset_index=0, 
            rebalance_tolerance=0.001 
        ) 
    # Set initial position 
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
             InformationCard(name="rebalancing", payload={})) 
        ) 
        t += stopping_interval 
    # Information generator 
    information_generator = InformationGenerator( 
        option_params=option_params, 
        lookback_window=10, 
        gamma_threshold=0.015 
    ) if use_information else None 
    # Event engine 
    engine = EventEngine(market, stopping_events, information_generator) 
    # Execution 
    execution_engine = ExecutionEngine() 
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
    # Replication error 
    replication_error = VT - option_payoff 
    # Additional metrics 
    result = { 
        'replication_error': replication_error, 
        'trade_count': agent.trade_count, 
        'total_costs': agent.total_transaction_costs, 
        'final_value': VT 
    }
    # Add adaptation summary if information-aware 
    if use_information and hasattr(agent, 'get_adaptation_summary'): 
        result['adaptation_summary'] = agent.get_adaptation_summary() 
    return result 
def run_experiment(): 
    """Compare information-aware vs naive strategies""" 
    stopping_intervals = [0.05, 0.1, 0.2] 
    n_paths = 1000 
    print("\n" + "="*70) 
    print("EXPERIMENT 3: Information-Aware vs Naive Delta Hedging") 
    print("="*70) 
    print(f"Number of paths: {n_paths}") 
    print(f"Option parameters: S0=100, K=100, r=0.05, sigma=0.2, T=1.0") 
    print("="*70 + "\n") 
    results = {} 
    for interval in stopping_intervals: 
        print(f"\nStopping Interval Δt = {interval:.3f}") 
        print("-" * 70) 
        # Naive strategy 
        naive_errors = [] 
        for i in range(n_paths): 
            result = run_single_path_information_aware( 
                stopping_interval=interval, 
                use_information=False, 
                seed=i 
            ) 
            naive_errors.append(result['replication_error']) 
        # Information-aware strategy 
        info_errors = [] 
        info_adaptations = [] 
        for i in range(n_paths): 
            result = run_single_path_information_aware( 
                stopping_interval=interval, 
                use_information=True, 
                seed=i 
            ) 
            info_errors.append(result['replication_error']) 
            if 'adaptation_summary' in result: 
                info_adaptations.append(result['adaptation_summary']) 
        naive_errors = np.array(naive_errors) 
        info_errors = np.array(info_errors) 
        # Statistics 
        print("Naive Strategy:") 
        print(f"  Mean error:        {np.mean(naive_errors):>8.4f}") 
        print(f"  Std error:         {np.std(naive_errors):>8.4f}") 
        print(f"  P(loss):           {np.mean(naive_errors < 0):>8.3f}") 
        print("\nInformation-Aware Strategy:") 
        print(f"  Mean error:        {np.mean(info_errors):>8.4f}") 
        print(f"  Std error:         {np.std(info_errors):>8.4f}") 
        print(f"  P(loss):           {np.mean(info_errors < 0):>8.3f}") 
        # Improvement 
        mean_improvement = np.mean(naive_errors) - np.mean(info_errors) 
        std_improvement = np.std(naive_errors) - np.std(info_errors) 
        print("\nImprovement:") 
        print(f"  Mean error:        {mean_improvement:>8.4f} ({mean_improvement/np.abs(np.mean(naive_errors))*100:.1f}%)")
        print(f"  Std error:         {std_improvement:>8.4f} ({std_improvement/np.std(naive_errors)*100:.1f}%)") 
        results[interval] = { 
            'naive': naive_errors, 
            'information_aware': info_errors, 
            'adaptations': info_adaptations 
        } 
    return results 
if __name__ == "__main__": 
    results = run_experiment()