import numpy as np
from market.substrate import MarketSubstrate


def test_discounted_martingale():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    mu = r
    T = 1.0
    dt = 0.01
    N = 5000

    discounted_final = []

    for i in range(N):
        m = MarketSubstrate(0.0, [S0], mu, sigma, seed=i)
        for _ in range(int(T / dt)):
            m.advance_time(dt)
        discounted_final.append(np.exp(-r * T) * m.prices[0])

    discounted_final = np.array(discounted_final)

    assert abs(discounted_final.mean() - S0) < 1.0
