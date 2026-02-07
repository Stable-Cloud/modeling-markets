# analysis/error_decomposition.py

import numpy as np
from options.black_scholes import black_scholes_gamma, black_scholes_vega


def decompose_hedging_error(portfolio_history, market_history, option_params, realized_vol=None):
    """
    Decompose total hedging error into components.
    
    The total hedging error can be attributed to:
    1. Gamma error: Error from discrete rebalancing (convexity)
    2. Vega error: Error from volatility misestimation
    3. Transaction cost error: Direct cost of trading
    
    Parameters
    ----------
    portfolio_history : list of dict
        Portfolio tracker history with 'time', 'value', 'cash', 'positions'
    market_history : list of dict
        Market state history with 'time', 'prices'
    option_params : dict
        Option parameters: {'K', 'r', 'sigma', 'T'}
    realized_vol : float, optional
        Realized volatility. If None, computed from price path
    
    Returns
    -------
    dict
        Dictionary with error components:
        - 'gamma_error': Estimated gamma error
        - 'vega_error': Estimated vega error (if realized_vol provided)
        - 'transaction_cost_error': Total transaction costs
        - 'total_error': Sum of components
        - 'unexplained_error': Residual
    """
    K = option_params['K']
    r = option_params['r']
    sigma = option_params['sigma']
    T = option_params['T']
    
    # Extract price path
    times = [state['time'] for state in market_history]
    prices = [state['prices'][0] for state in market_history]
    
    # Compute gamma error (integral of 0.5 * Gamma * (dS)^2)
    gamma_error = 0.0
    
    for i in range(1, len(times)):
        t = times[i-1]
        S = prices[i-1]
        dS = prices[i] - prices[i-1]
        dt = times[i] - times[i-1]
        
        # Skip if past maturity
        if t >= T:
            continue
        
        # Compute gamma at this point
        gamma = black_scholes_gamma(S, K, r, sigma, T, t)
        
        # Gamma error contribution: 0.5 * Gamma * (dS)^2
        gamma_error += 0.5 * gamma * (dS ** 2)
    
    # Compute vega error if realized volatility provided
    vega_error = 0.0
    if realized_vol is not None:
        # Compute average vega over the path
        avg_vega = 0.0
        count = 0
        
        for i in range(len(times)):
            t = times[i]
            S = prices[i]
            
            if t >= T:
                continue
            
            vega = black_scholes_vega(S, K, r, sigma, T, t)
            avg_vega += vega
            count += 1
        
        if count > 0:
            avg_vega /= count
            # Vega error: average vega * (realized_vol - implied_vol)
            vega_error = avg_vega * (realized_vol - sigma)
    
    # Transaction cost error (from portfolio history)
    # This should be tracked in the agent, but we can estimate from capital changes
    transaction_cost_error = 0.0
    if len(portfolio_history) > 0 and 'total_transaction_costs' in portfolio_history[-1]:
        # If available directly
        pass
    
    # Total decomposition
    decomposition = {
        'gamma_error': gamma_error,
        'vega_error': vega_error,
        'transaction_cost_error': transaction_cost_error,
        'explained_error': gamma_error + vega_error + transaction_cost_error
    }
    
    return decomposition


def compute_realized_volatility(prices, times):
    """
    Compute realized volatility from a price path.
    
    Uses log returns and annualizes based on time intervals.
    
    Parameters
    ----------
    prices : array_like
        Price path
    times : array_like
        Time points corresponding to prices
    
    Returns
    -------
    float
        Annualized realized volatility
    """
    prices = np.array(prices)
    times = np.array(times)
    
    if len(prices) < 2:
        return 0.0
    
    # Compute log returns
    log_returns = np.diff(np.log(prices))
    
    # Compute time intervals
    dt_values = np.diff(times)
    
    # Variance of log returns
    var_returns = np.var(log_returns)
    
    # Average time interval
    avg_dt = np.mean(dt_values)
    
    # Annualize (assuming dt is in years)
    if avg_dt > 0:
        annualized_vol = np.sqrt(var_returns / avg_dt)
    else:
        annualized_vol = 0.0
    
    return annualized_vol


def analyze_error_sources(simulation_results):
    """
    Analyze sources of hedging error across multiple paths.
    
    Parameters
    ----------
    simulation_results : list of dict
        List of simulation results, each containing:
        - 'portfolio_history'
        - 'market_history'
        - 'option_params'
        - 'replication_error'
    
    Returns
    -------
    dict
        Aggregated error decomposition statistics
    """
    gamma_errors = []
    vega_errors = []
    total_errors = []
    
    for result in simulation_results:
        decomp = decompose_hedging_error(
            result['portfolio_history'],
            result['market_history'],
            result['option_params']
        )
        
        gamma_errors.append(decomp['gamma_error'])
        vega_errors.append(decomp['vega_error'])
        total_errors.append(result['replication_error'])
    
    return {
        'mean_gamma_error': np.mean(gamma_errors),
        'std_gamma_error': np.std(gamma_errors),
        'mean_vega_error': np.mean(vega_errors),
        'std_vega_error': np.std(vega_errors),
        'mean_total_error': np.mean(total_errors),
        'std_total_error': np.std(total_errors),
        'gamma_contribution': np.mean(gamma_errors) / np.mean(np.abs(total_errors)) if np.mean(np.abs(total_errors)) > 0 else 0
    }
