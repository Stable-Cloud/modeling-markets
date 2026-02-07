# analysis/risk_metrics.py

import numpy as np


def compute_var(errors, confidence_level=0.95):
    """
    Compute Value at Risk (VaR) at given confidence level.
    
    VaR is the maximum loss not exceeded with a given confidence level.
    For hedging errors, this represents the worst-case shortfall.
    
    Parameters
    ----------
    errors : array_like
        Array of replication errors (V_T - Payoff)
    confidence_level : float, optional
        Confidence level (default: 0.95 for 95% VaR)
    
    Returns
    -------
    float
        VaR value (negative indicates potential loss)
    
    Examples
    --------
    >>> errors = np.random.normal(0, 1, 1000)
    >>> var_95 = compute_var(errors, 0.95)
    """
    errors = np.array(errors)
    percentile = (1 - confidence_level) * 100
    var = np.percentile(errors, percentile)
    return var


def compute_cvar(errors, confidence_level=0.95):
    """
    Compute Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    It provides a better measure of tail risk than VaR.
    
    Parameters
    ----------
    errors : array_like
        Array of replication errors
    confidence_level : float, optional
        Confidence level (default: 0.95)
    
    Returns
    -------
    float
        CVaR value (expected loss in the tail)
    
    Examples
    --------
    >>> errors = np.random.normal(0, 1, 1000)
    >>> cvar_95 = compute_cvar(errors, 0.95)
    """
    errors = np.array(errors)
    var = compute_var(errors, confidence_level)
    
    # CVaR is the mean of errors below VaR
    tail_errors = errors[errors <= var]
    
    if len(tail_errors) > 0:
        cvar = np.mean(tail_errors)
    else:
        cvar = var
    
    return cvar


def compute_shortfall_probability(errors, threshold=0):
    """
    Compute probability that error falls below threshold.
    
    For hedging, this is P(V_T < Payoff), the probability of
    under-hedging (portfolio value insufficient to cover payoff).
    
    Parameters
    ----------
    errors : array_like
        Array of replication errors
    threshold : float, optional
        Threshold value (default: 0)
    
    Returns
    -------
    float
        Probability of shortfall
    
    Examples
    --------
    >>> errors = np.random.normal(0, 1, 1000)
    >>> prob_loss = compute_shortfall_probability(errors, 0)
    """
    errors = np.array(errors)
    return np.mean(errors < threshold)


def compute_maximum_drawdown(portfolio_values):
    """
    Compute maximum drawdown of portfolio value.
    
    Maximum drawdown is the largest peak-to-trough decline.
    
    Parameters
    ----------
    portfolio_values : array_like
        Time series of portfolio values
    
    Returns
    -------
    dict
        Dictionary with:
        - 'max_drawdown': Maximum drawdown value
        - 'peak_idx': Index of peak before drawdown
        - 'trough_idx': Index of trough
        - 'drawdown_pct': Drawdown as percentage of peak
    
    Examples
    --------
    >>> values = [100, 110, 105, 95, 100, 120]
    >>> dd = compute_maximum_drawdown(values)
    >>> dd['max_drawdown']
    15.0
    """
    values = np.array(portfolio_values)
    
    if len(values) == 0:
        return {
            'max_drawdown': 0.0,
            'peak_idx': 0,
            'trough_idx': 0,
            'drawdown_pct': 0.0
        }
    
    # Compute running maximum
    running_max = np.maximum.accumulate(values)
    
    # Compute drawdown at each point
    drawdowns = running_max - values
    
    # Find maximum drawdown
    max_dd_idx = np.argmax(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    
    # Find the peak before this drawdown
    peak_idx = np.argmax(values[:max_dd_idx+1]) if max_dd_idx > 0 else 0
    peak_value = values[peak_idx]
    
    # Drawdown percentage
    dd_pct = (max_dd / peak_value * 100) if peak_value > 0 else 0.0
    
    return {
        'max_drawdown': max_dd,
        'peak_idx': peak_idx,
        'trough_idx': max_dd_idx,
        'drawdown_pct': dd_pct
    }


def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Compute Sharpe ratio of returns.
    
    Sharpe ratio = (mean return - risk-free rate) / std(returns)
    
    Parameters
    ----------
    returns : array_like
        Array of returns
    risk_free_rate : float, optional
        Risk-free rate (default: 0.0)
    
    Returns
    -------
    float
        Sharpe ratio
    """
    returns = np.array(returns)
    
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_return = np.mean(returns) - risk_free_rate
    sharpe = excess_return / np.std(returns)
    
    return sharpe


def compute_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Compute Sortino ratio of returns.
    
    Similar to Sharpe ratio but only penalizes downside volatility.
    
    Parameters
    ----------
    returns : array_like
        Array of returns
    risk_free_rate : float, optional
        Risk-free rate (default: 0.0)
    target_return : float, optional
        Target return threshold (default: 0.0)
    
    Returns
    -------
    float
        Sortino ratio
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    # Downside returns (below target)
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(returns) > risk_free_rate else 0.0
    
    # Downside deviation
    downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    if downside_dev == 0:
        return 0.0
    
    excess_return = np.mean(returns) - risk_free_rate
    sortino = excess_return / downside_dev
    
    return sortino


def generate_risk_report(errors, portfolio_history=None):
    """
    Generate comprehensive risk report.
    
    Parameters
    ----------
    errors : array_like
        Array of replication errors
    portfolio_history : list of dict, optional
        Portfolio value history for drawdown analysis
    
    Returns
    -------
    dict
        Dictionary with all risk metrics
    """
    errors = np.array(errors)
    
    report = {
        # Basic statistics
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        
        # Percentiles
        'percentile_1': np.percentile(errors, 1),
        'percentile_5': np.percentile(errors, 5),
        'percentile_25': np.percentile(errors, 25),
        'percentile_75': np.percentile(errors, 75),
        'percentile_95': np.percentile(errors, 95),
        'percentile_99': np.percentile(errors, 99),
        
        # VaR and CVaR
        'var_90': compute_var(errors, 0.90),
        'var_95': compute_var(errors, 0.95),
        'var_99': compute_var(errors, 0.99),
        'cvar_90': compute_cvar(errors, 0.90),
        'cvar_95': compute_cvar(errors, 0.95),
        'cvar_99': compute_cvar(errors, 0.99),
        
        # Shortfall probabilities
        'prob_loss': compute_shortfall_probability(errors, 0),
        'prob_loss_1pct': compute_shortfall_probability(errors, -0.01),
        'prob_loss_5pct': compute_shortfall_probability(errors, -0.05),
        
        # Distribution shape
        'skewness': float(np.mean(((errors - np.mean(errors)) / np.std(errors)) ** 3)),
        'kurtosis': float(np.mean(((errors - np.mean(errors)) / np.std(errors)) ** 4)),
    }
    
    # Add drawdown if portfolio history provided
    if portfolio_history is not None:
        values = [h['value'] for h in portfolio_history]
        dd = compute_maximum_drawdown(values)
        report['max_drawdown'] = dd['max_drawdown']
        report['max_drawdown_pct'] = dd['drawdown_pct']
    
    return report


def print_risk_report(report):
    """
    Print formatted risk report.
    
    Parameters
    ----------
    report : dict
        Risk report from generate_risk_report()
    """
    print("\n" + "="*70)
    print("RISK METRICS REPORT")
    print("="*70)
    
    print("\nBasic Statistics:")
    print(f"  Mean error:           {report['mean_error']:>10.4f}")
    print(f"  Median error:         {report['median_error']:>10.4f}")
    print(f"  Std deviation:        {report['std_error']:>10.4f}")
    print(f"  Min error:            {report['min_error']:>10.4f}")
    print(f"  Max error:            {report['max_error']:>10.4f}")
    
    print("\nPercentiles:")
    print(f"  1st percentile:       {report['percentile_1']:>10.4f}")
    print(f"  5th percentile:       {report['percentile_5']:>10.4f}")
    print(f"  25th percentile:      {report['percentile_25']:>10.4f}")
    print(f"  75th percentile:      {report['percentile_75']:>10.4f}")
    print(f"  95th percentile:      {report['percentile_95']:>10.4f}")
    print(f"  99th percentile:      {report['percentile_99']:>10.4f}")
    
    print("\nValue at Risk (VaR):")
    print(f"  VaR 90%:              {report['var_90']:>10.4f}")
    print(f"  VaR 95%:              {report['var_95']:>10.4f}")
    print(f"  VaR 99%:              {report['var_99']:>10.4f}")
    
    print("\nConditional VaR (CVaR / Expected Shortfall):")
    print(f"  CVaR 90%:             {report['cvar_90']:>10.4f}")
    print(f"  CVaR 95%:             {report['cvar_95']:>10.4f}")
    print(f"  CVaR 99%:             {report['cvar_99']:>10.4f}")
    
    print("\nShortfall Probabilities:")
    print(f"  P(error < 0):         {report['prob_loss']:>10.3f}")
    print(f"  P(error < -1%):       {report['prob_loss_1pct']:>10.3f}")
    print(f"  P(error < -5%):       {report['prob_loss_5pct']:>10.3f}")
    
    print("\nDistribution Shape:")
    print(f"  Skewness:             {report['skewness']:>10.4f}")
    print(f"  Kurtosis:             {report['kurtosis']:>10.4f}")
    
    if 'max_drawdown' in report:
        print("\nDrawdown Analysis:")
        print(f"  Max drawdown:         {report['max_drawdown']:>10.4f}")
        print(f"  Max drawdown %:       {report['max_drawdown_pct']:>10.2f}%")
    
    print("\n" + "="*70 + "\n")
