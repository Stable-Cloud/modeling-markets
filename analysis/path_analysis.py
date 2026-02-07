# analysis/path_analysis.py

import numpy as np
from options.black_scholes import black_scholes_gamma


def identify_difficult_paths(simulation_results, threshold_percentile=90):
    """
    Identify paths with largest hedging errors.
    
    Parameters
    ----------
    simulation_results : list of dict
        List of simulation results with 'replication_error' key
    threshold_percentile : float, optional
        Percentile threshold for "difficult" paths (default: 90)
    
    Returns
    -------
    dict
        Dictionary with:
        - 'difficult_indices': Indices of difficult paths
        - 'threshold': Error threshold value
        - 'characteristics': Path characteristics for difficult paths
    """
    errors = np.array([r['replication_error'] for r in simulation_results])
    
    # Compute threshold (absolute error)
    abs_errors = np.abs(errors)
    threshold = np.percentile(abs_errors, threshold_percentile)
    
    # Identify difficult paths
    difficult_indices = np.where(abs_errors >= threshold)[0]
    
    return {
        'difficult_indices': difficult_indices,
        'threshold': threshold,
        'count': len(difficult_indices),
        'percentage': len(difficult_indices) / len(errors) * 100
    }


def extract_path_characteristics(market_history, option_params):
    """
    Extract characteristics from a price path.
    
    Parameters
    ----------
    market_history : list of dict
        Market state history with 'time' and 'prices'
    option_params : dict
        Option parameters: {'K', 'r', 'sigma', 'T'}
    
    Returns
    -------
    dict
        Dictionary with path characteristics:
        - 'max_price': Maximum price reached
        - 'min_price': Minimum price reached
        - 'price_range': max - min
        - 'final_price': Terminal price
        - 'total_variation': Sum of absolute price changes
        - 'num_strike_crosses': Number of times price crossed strike
        - 'max_gamma': Maximum gamma exposure
        - 'avg_gamma': Average gamma exposure
        - 'time_itm': Fraction of time in-the-money
    """
    K = option_params['K']
    r = option_params['r']
    sigma = option_params['sigma']
    T = option_params['T']
    
    times = np.array([s['time'] for s in market_history])
    prices = np.array([s['prices'][0] for s in market_history])
    
    # Basic price statistics
    max_price = np.max(prices)
    min_price = np.min(prices)
    price_range = max_price - min_price
    final_price = prices[-1]
    
    # Total variation
    total_variation = np.sum(np.abs(np.diff(prices)))
    
    # Strike crosses
    above_strike = prices > K
    num_crosses = np.sum(np.abs(np.diff(above_strike.astype(int))))
    
    # Gamma statistics
    gammas = []
    for i, (t, S) in enumerate(zip(times, prices)):
        if t < T:
            gamma = black_scholes_gamma(S, K, r, sigma, T, t)
            gammas.append(gamma)
    
    max_gamma = np.max(gammas) if gammas else 0.0
    avg_gamma = np.mean(gammas) if gammas else 0.0
    
    # Time in-the-money
    time_itm = np.mean(prices > K)
    
    # Realized volatility
    if len(prices) > 1:
        log_returns = np.diff(np.log(prices))
        dt_avg = np.mean(np.diff(times))
        realized_vol = np.std(log_returns) / np.sqrt(dt_avg) if dt_avg > 0 else 0.0
    else:
        realized_vol = 0.0
    
    return {
        'max_price': max_price,
        'min_price': min_price,
        'price_range': price_range,
        'final_price': final_price,
        'total_variation': total_variation,
        'num_strike_crosses': num_crosses,
        'max_gamma': max_gamma,
        'avg_gamma': avg_gamma,
        'time_itm': time_itm,
        'realized_vol': realized_vol
    }


def correlate_characteristics_with_errors(simulation_results):
    """
    Correlate path characteristics with hedging errors.
    
    Identifies which path features predict large errors.
    
    Parameters
    ----------
    simulation_results : list of dict
        Simulation results with 'market_history', 'option_params', 'replication_error'
    
    Returns
    -------
    dict
        Correlation coefficients between characteristics and errors
    """
    errors = []
    characteristics = {
        'max_price': [],
        'min_price': [],
        'price_range': [],
        'total_variation': [],
        'num_strike_crosses': [],
        'max_gamma': [],
        'avg_gamma': [],
        'time_itm': [],
        'realized_vol': []
    }
    
    for result in simulation_results:
        errors.append(result['replication_error'])
        
        chars = extract_path_characteristics(
            result['market_history'],
            result['option_params']
        )
        
        for key in characteristics:
            characteristics[key].append(chars[key])
    
    errors = np.array(errors)
    abs_errors = np.abs(errors)
    
    # Compute correlations
    correlations = {}
    for key, values in characteristics.items():
        values = np.array(values)
        
        # Correlation with absolute error
        if np.std(values) > 0 and np.std(abs_errors) > 0:
            corr = np.corrcoef(values, abs_errors)[0, 1]
            correlations[key] = corr
        else:
            correlations[key] = 0.0
    
    # Sort by absolute correlation
    sorted_correlations = dict(
        sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    
    return sorted_correlations


def analyze_error_by_moneyness(simulation_results, num_bins=5):
    """
    Analyze hedging error by option moneyness (S/K).
    
    Parameters
    ----------
    simulation_results : list of dict
        Simulation results
    num_bins : int, optional
        Number of moneyness bins (default: 5)
    
    Returns
    -------
    dict
        Error statistics by moneyness bin
    """
    moneyness_values = []
    errors = []
    
    for result in simulation_results:
        # Terminal moneyness
        final_price = result['market_history'][-1]['prices'][0]
        K = result['option_params']['K']
        moneyness = final_price / K
        
        moneyness_values.append(moneyness)
        errors.append(result['replication_error'])
    
    moneyness_values = np.array(moneyness_values)
    errors = np.array(errors)
    
    # Create bins
    bins = np.linspace(np.min(moneyness_values), np.max(moneyness_values), num_bins + 1)
    bin_indices = np.digitize(moneyness_values, bins) - 1
    
    # Compute statistics per bin
    results_by_bin = {}
    for i in range(num_bins):
        mask = bin_indices == i
        bin_errors = errors[mask]
        
        if len(bin_errors) > 0:
            results_by_bin[i] = {
                'moneyness_range': (bins[i], bins[i+1]),
                'count': len(bin_errors),
                'mean_error': np.mean(bin_errors),
                'std_error': np.std(bin_errors),
                'median_error': np.median(bin_errors)
            }
    
    return results_by_bin


def print_path_analysis_report(correlations, moneyness_analysis=None):
    """
    Print formatted path analysis report.
    
    Parameters
    ----------
    correlations : dict
        Correlation coefficients from correlate_characteristics_with_errors()
    moneyness_analysis : dict, optional
        Results from analyze_error_by_moneyness()
    """
    print("\n" + "="*70)
    print("PATH ANALYSIS REPORT")
    print("="*70)
    
    print("\nCorrelation with Absolute Hedging Error:")
    print("(Higher absolute correlation = stronger predictor of large errors)")
    print("-" * 70)
    
    for feature, corr in correlations.items():
        feature_name = feature.replace('_', ' ').title()
        print(f"  {feature_name:<25} {corr:>8.4f}")
    
    if moneyness_analysis:
        print("\n" + "="*70)
        print("Error by Terminal Moneyness (S_T / K):")
        print("-" * 70)
        
        for bin_idx, stats in moneyness_analysis.items():
            m_low, m_high = stats['moneyness_range']
            print(f"\nMoneyness [{m_low:.2f}, {m_high:.2f}]:")
            print(f"  Count:        {stats['count']:>6}")
            print(f"  Mean error:   {stats['mean_error']:>8.4f}")
            print(f"  Std error:    {stats['std_error']:>8.4f}")
            print(f"  Median error: {stats['median_error']:>8.4f}")
    
    print("\n" + "="*70 + "\n")
