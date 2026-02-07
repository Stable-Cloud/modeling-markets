# options/black_scholes.py

import math
from math import log, sqrt, exp, pi
from scipy.stats import norm
import numpy as np


def black_scholes_call_price(S0, K, r, sigma, T):
    """
    Compute Black-Scholes price for European call option.
    
    Parameters
    ----------
    S0 : float
        Current stock price (must be positive)
    K : float
        Strike price (must be positive)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Time to maturity in years (must be non-negative)
    
    Returns
    -------
    float
        Call option price
    
    Examples
    --------
    >>> black_scholes_call_price(100, 100, 0.05, 0.2, 1.0)
    10.450583572185565
    """
    # Input validation
    if S0 <= 0:
        raise ValueError(f"Stock price S0 must be positive. Got: {S0}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive. Got: {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive. Got: {sigma}")
    if T < 0:
        raise ValueError(f"Time to maturity T must be non-negative. Got: {T}")
    
    # Edge case: at maturity
    if T == 0:
        return max(S0 - K, 0.0)
    
    # Edge case: zero volatility
    if sigma == 0:
        return max(exp(-r * T) * (S0 * exp(r * T) - K), 0.0)
    
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def black_scholes_delta(S, K, r, sigma, T, t=0.0):
    """
    Compute delta (∂C/∂S) for European call option.
    
    Delta represents the rate of change of option price with respect to
    the underlying asset price. For a call option, delta ∈ [0, 1].
    
    Parameters
    ----------
    S : float
        Current stock price (must be positive)
    K : float
        Strike price (must be positive)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Maturity time in years
    t : float, optional
        Current time in years (default: 0.0)
    
    Returns
    -------
    float
        Delta value in [0, 1]
    
    Examples
    --------
    >>> black_scholes_delta(100, 100, 0.05, 0.2, 1.0, 0.0)
    0.6368306500170766
    """
    # Input validation
    if S <= 0:
        raise ValueError(f"Stock price S must be positive. Got: {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive. Got: {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive. Got: {sigma}")
    if T < t:
        raise ValueError(f"Maturity T must be >= current time t. Got T={T}, t={t}")
    
    tau = T - t  # Time to maturity
    
    # Edge case: at maturity
    if tau == 0:
        return 1.0 if S > K else 0.0
    
    # Edge case: zero volatility
    if sigma == 0:
        return 1.0 if S > K * exp(-r * tau) else 0.0
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    return norm.cdf(d1)


def black_scholes_gamma(S, K, r, sigma, T, t=0.0):
    """
    Compute gamma (∂²C/∂S²) for European call option.
    
    Gamma represents the rate of change of delta with respect to the
    underlying asset price. It measures the convexity of the option.
    
    Parameters
    ----------
    S : float
        Current stock price (must be positive)
    K : float
        Strike price (must be positive)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Maturity time in years
    t : float, optional
        Current time in years (default: 0.0)
    
    Returns
    -------
    float
        Gamma value (always non-negative)
    
    Examples
    --------
    >>> black_scholes_gamma(100, 100, 0.05, 0.2, 1.0, 0.0)
    0.018683883789920413
    """
    # Input validation
    if S <= 0:
        raise ValueError(f"Stock price S must be positive. Got: {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive. Got: {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive. Got: {sigma}")
    if T < t:
        raise ValueError(f"Maturity T must be >= current time t. Got T={T}, t={t}")
    
    tau = T - t  # Time to maturity
    
    # Edge case: at maturity
    if tau == 0:
        return 0.0
    
    # Edge case: zero volatility
    if sigma == 0:
        return 0.0
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    # Standard normal PDF: φ(x) = (1/√(2π)) * exp(-x²/2)
    phi_d1 = exp(-0.5 * d1**2) / sqrt(2 * pi)
    return phi_d1 / (S * sigma * sqrt(tau))


def black_scholes_vega(S, K, r, sigma, T, t=0.0):
    """
    Compute vega (∂C/∂σ) for European call option.
    
    Vega represents the sensitivity of option price to changes in volatility.
    Note: Vega is typically expressed per 1% change in volatility.
    
    Parameters
    ----------
    S : float
        Current stock price (must be positive)
    K : float
        Strike price (must be positive)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Maturity time in years
    t : float, optional
        Current time in years (default: 0.0)
    
    Returns
    -------
    float
        Vega value (sensitivity per unit change in sigma)
    
    Examples
    --------
    >>> black_scholes_vega(100, 100, 0.05, 0.2, 1.0, 0.0)
    37.36776757984083
    """
    # Input validation
    if S <= 0:
        raise ValueError(f"Stock price S must be positive. Got: {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive. Got: {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive. Got: {sigma}")
    if T < t:
        raise ValueError(f"Maturity T must be >= current time t. Got T={T}, t={t}")
    
    tau = T - t  # Time to maturity
    
    # Edge case: at maturity
    if tau == 0:
        return 0.0
    
    # Edge case: zero volatility
    if sigma == 0:
        return 0.0
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    # Standard normal PDF: φ(x) = (1/√(2π)) * exp(-x²/2)
    phi_d1 = exp(-0.5 * d1**2) / sqrt(2 * pi)
    return S * phi_d1 * sqrt(tau)


def black_scholes_theta(S, K, r, sigma, T, t=0.0):
    """
    Compute theta (∂C/∂t) for European call option.
    
    Theta represents the time decay of the option value. By convention,
    theta is often expressed as negative (option loses value as time passes).
    This function returns -∂C/∂t (the rate of time decay).
    
    Parameters
    ----------
    S : float
        Current stock price (must be positive)
    K : float
        Strike price (must be positive)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized, must be positive)
    T : float
        Maturity time in years
    t : float, optional
        Current time in years (default: 0.0)
    
    Returns
    -------
    float
        Theta value (typically negative, representing time decay)
    
    Examples
    --------
    >>> black_scholes_theta(100, 100, 0.05, 0.2, 1.0, 0.0)
    -6.414101933621063
    """
    # Input validation
    if S <= 0:
        raise ValueError(f"Stock price S must be positive. Got: {S}")
    if K <= 0:
        raise ValueError(f"Strike price K must be positive. Got: {K}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive. Got: {sigma}")
    if T < t:
        raise ValueError(f"Maturity T must be >= current time t. Got T={T}, t={t}")
    
    tau = T - t  # Time to maturity
    
    # Edge case: at maturity
    if tau == 0:
        return 0.0
    
    # Edge case: zero volatility
    if sigma == 0:
        return 0.0
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    
    # Standard normal PDF: φ(x) = (1/√(2π)) * exp(-x²/2)
    phi_d1 = exp(-0.5 * d1**2) / sqrt(2 * pi)
    
    term1 = -(S * phi_d1 * sigma) / (2 * sqrt(tau))
    term2 = -r * K * exp(-r * tau) * norm.cdf(d2)
    
    return term1 + term2
