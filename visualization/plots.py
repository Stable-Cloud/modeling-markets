# visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_single_path_detailed(market_history, portfolio_history, option_params, agent=None):
    """
    Create detailed multi-panel plot for a single path.
    
    Parameters
    ----------
    market_history : list of dict
        Market state history
    portfolio_history : list of dict
        Portfolio tracker history
    option_params : dict
        Option parameters
    agent : Agent, optional
        Agent instance for additional info
    """
    from options.black_scholes import black_scholes_delta, black_scholes_gamma, black_scholes_call_price
    
    # Extract data
    times = [s['time'] for s in market_history]
    prices = [s['prices'][0] for s in market_history]
    
    portfolio_times = [h['time'] for h in portfolio_history]
    portfolio_values = [h['value'] for h in portfolio_history]
    cash_values = [h['cash'] for h in portfolio_history]
    positions = [h['positions'].get(0, 0.0) for h in portfolio_history]
    
    K = option_params['K']
    r = option_params['r']
    sigma = option_params['sigma']
    T = option_params['T']
    
    # Compute theoretical option values and Greeks along path
    option_values = []
    deltas = []
    gammas = []
    
    for t, S in zip(times, prices):
        if t < T:
            C = black_scholes_call_price(S, K, r, sigma, T - t)
            delta = black_scholes_delta(S, K, r, sigma, T, t)
            gamma = black_scholes_gamma(S, K, r, sigma, T, t)
        else:
            C = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
            gamma = 0.0
        
        option_values.append(C)
        deltas.append(delta)
        gammas.append(gamma)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Stock Price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, prices, 'b-', linewidth=2, label='Stock Price')
    ax1.axhline(K, color='r', linestyle='--', linewidth=1, label=f'Strike K={K}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Stock Price Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Portfolio Value vs Option Value
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(portfolio_times, portfolio_values, 'g-', linewidth=2, label='Portfolio Value')
    ax2.plot(times, option_values, 'r--', linewidth=2, label='Option Value (BS)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Portfolio vs Option Value', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Delta Position
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(portfolio_times, positions, 'purple', linewidth=2, label='Actual Position')
    ax3.plot(times, deltas, 'orange', linestyle='--', linewidth=2, label='Target Delta')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position (shares)')
    ax3.set_title('Delta Hedging Position', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Gamma Exposure
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(times, gammas, 'brown', linewidth=2)
    ax4.fill_between(times, 0, gammas, alpha=0.3, color='brown')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Gamma')
    ax4.set_title('Gamma Exposure Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Cash Position
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(portfolio_times, cash_values, 'teal', linewidth=2)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Cash')
    ax5.set_title('Cash Position', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Path Analysis', fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def plot_error_distribution_comparison(results_dict, title="Replication Error Distribution"):
    """
    Compare error distributions across scenarios.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping scenario names to error arrays
    title : str, optional
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    for label, errors in results_dict.items():
        ax1.hist(errors, bins=50, alpha=0.5, density=True, label=label)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Replication Error')
    ax1.set_ylabel('Density')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2.boxplot([errors for errors in results_dict.values()],
                labels=list(results_dict.keys()))
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel('Replication Error')
    ax2.set_title('Error Distribution (Box Plot)')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_convergence_analysis(intervals, errors_dict):
    """
    Plot convergence analysis showing error vs stopping interval.
    
    Parameters
    ----------
    intervals : array_like
        Stopping intervals
    errors_dict : dict
        Dictionary with 'mean', 'std', 'var' arrays
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean error vs interval
    ax1.plot(intervals, errors_dict['mean'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Stopping Interval Δt')
    ax1.set_ylabel('Mean Replication Error')
    ax1.set_title('Mean Error vs Rebalancing Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Variance on log-log scale
    ax2.loglog(intervals, errors_dict['var'], 'o-', linewidth=2, markersize=8, label='Empirical')
    
    # Theoretical O(Δt) line
    theoretical = errors_dict['var'][0] * (np.array(intervals) / intervals[0])
    ax2.loglog(intervals, theoretical, '--', linewidth=2, label='O(Δt) theoretical')
    
    ax2.set_xlabel('Stopping Interval Δt')
    ax2.set_ylabel('Variance of Error')
    ax2.set_title('Error Variance Scaling (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_transaction_cost_impact(cost_rates, intervals, results):
    """
    Plot impact of transaction costs on hedging performance.
    
    Parameters
    ----------
    cost_rates : array_like
        Transaction cost rates (bps)
    intervals : array_like
        Stopping intervals
    results : dict
        Nested dict: {cost_rate: {interval: {'mean_error', 'mean_cost'}}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean error vs interval for different cost rates
    for cost_bps in cost_rates:
        mean_errors = [results[cost_bps][interval]['mean_error'] 
                      for interval in intervals]
        ax1.plot(intervals, mean_errors, 'o-', label=f'{cost_bps} bps', 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Stopping Interval Δt')
    ax1.set_ylabel('Mean Replication Error')
    ax1.set_title('Error vs Rebalancing Frequency (Different Cost Rates)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Total cost (error + transaction costs) vs interval
    for cost_bps in cost_rates:
        total_costs = [results[cost_bps][interval]['mean_error'] + 
                      results[cost_bps][interval]['mean_cost']
                      for interval in intervals]
        ax2.plot(intervals, total_costs, 'o-', label=f'{cost_bps} bps',
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('Stopping Interval Δt')
    ax2.set_ylabel('Total Cost (Error + Transaction Costs)')
    ax2.set_title('Total Hedging Cost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_risk_metrics_dashboard(errors, portfolio_history=None):
    """
    Create dashboard with multiple risk visualizations.
    
    Parameters
    ----------
    errors : array_like
        Replication errors
    portfolio_history : list of dict, optional
        Portfolio history for additional analysis
    """
    from scipy import stats
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    errors = np.array(errors)
    
    # Panel 1: Histogram with VaR/CVaR
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add VaR lines
    var_95 = np.percentile(errors, 5)
    var_99 = np.percentile(errors, 1)
    ax1.axvline(var_95, color='orange', linestyle='--', linewidth=2, label='VaR 95%')
    ax1.axvline(var_99, color='red', linestyle='--', linewidth=2, label='VaR 99%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax1.set_xlabel('Replication Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution with VaR', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Cumulative Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax3.plot(sorted_errors, cumulative, linewidth=2)
    ax3.axvline(0, color='red', linestyle='--', linewidth=1)
    ax3.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax3.set_xlabel('Replication Error')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Portfolio value time series (if available)
    if portfolio_history:
        ax4 = fig.add_subplot(gs[1, 1])
        times = [h['time'] for h in portfolio_history]
        values = [h['value'] for h in portfolio_history]
        ax4.plot(times, values, linewidth=2, color='green')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Portfolio Value')
        ax4.set_title('Portfolio Value Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        # Error percentiles over time
        ax4 = fig.add_subplot(gs[1, 1])
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        values = [np.percentile(errors, p) for p in percentiles]
        ax4.barh(percentiles, values, color='steelblue', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=1)
        ax4.set_xlabel('Error Value')
        ax4.set_ylabel('Percentile')
        ax4.set_title('Error Percentiles', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Risk Metrics Dashboard', fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def set_plot_style():
    """Set consistent plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3
