#market/substrate.py
import numpy as np
import warnings


class MarketSubstrate:
    """
    Continuous-time market substrate implementing
    geometric Brownian motion.
    
    Uses Euler-Maruyama discretization: dS = μS dt + σS dW
    
    Parameters
    ----------
    initial_time : float
        Starting time for the simulation
    initial_prices : array_like
        Initial prices for each asset (must be positive)
    mu : float or array_like
        Drift coefficient(s)
    sigma : float or array_like
        Volatility coefficient(s) (must be non-negative)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, initial_time, initial_prices, mu, sigma, seed=None):
        self.time = float(initial_time)
        self.prices = np.array(initial_prices, dtype=float)
        self.mu = np.array(mu, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.rng = np.random.default_rng(seed)
        
        # Validation
        if np.any(self.prices <= 0):
            raise ValueError(
                f"All initial prices must be positive. Got: {self.prices}"
            )
        
        if np.any(self.sigma < 0):
            raise ValueError(
                f"Volatility (sigma) must be non-negative. Got: {self.sigma}"
            )

    def advance_time(self, dt):
        """
        Advance the market by time step dt using Euler-Maruyama scheme.
        
        Parameters
        ----------
        dt : float
            Time step (must be positive)
        
        Warnings
        --------
        Issues warning if dt is too large relative to volatility,
        which can cause numerical instability.
        """
        dt = float(dt)
        
        # Validate dt
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive. Got: {dt}")
        
        # Warn if dt is too large (rule of thumb: σ²dt < 0.1)
        max_sigma = np.max(self.sigma)
        if max_sigma**2 * dt > 0.1:
            warnings.warn(
                f"Large time step detected: σ²dt = {max_sigma**2 * dt:.4f} > 0.1. "
                f"This may cause numerical instability. Consider reducing dt.",
                RuntimeWarning
            )
        
        # Euler-Maruyama update
        Z = self.rng.standard_normal(size=self.prices.shape)
        drift = self.mu * self.prices * dt
        diffusion = self.sigma * self.prices * np.sqrt(dt) * Z
        self.prices = self.prices + drift + diffusion
        
        # Price floor protection (prevent negative prices)
        min_price = 1e-6
        if np.any(self.prices <= 0):
            warnings.warn(
                f"Negative or zero price detected. Clipping to {min_price}. "
                f"This indicates numerical instability - consider reducing dt.",
                RuntimeWarning
            )
            self.prices = np.maximum(self.prices, min_price)
        
        self.time += dt

    def get_state(self):
        """
        Get current market state.
        
        Returns
        -------
        dict
            Dictionary with 'time' and 'prices' keys
            
        Raises
        ------
        RuntimeError
            If state contains NaN or Inf values
        """
        # Validate state before returning
        if np.any(np.isnan(self.prices)) or np.any(np.isinf(self.prices)):
            raise RuntimeError(
                f"Corrupted market state detected at time {self.time}. "
                f"Prices contain NaN or Inf: {self.prices}"
            )
        
        return {
            "time": self.time,
            "prices": self.prices.copy()
        }
