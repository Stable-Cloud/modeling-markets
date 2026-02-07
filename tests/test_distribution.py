import numpy as np
from market.substrate import MarketSubstrate


def test_logprice_distribution():
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    N = 20000
    dt = 0.01

    log_prices = []

    for i in range(N):
        m = MarketSubstrate(0.0, [S0], mu, sigma, seed=i)
        for _ in range(int(T / dt)):
            m.advance_time(dt)
        log_prices.append(np.log(m.prices[0]))

    log_prices = np.array(log_prices)

    theoretical_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
    theoretical_var = sigma**2 * T

    assert abs(log_prices.mean() - theoretical_mean) < 0.05
    assert abs(log_prices.var() - theoretical_var) < 0.05
