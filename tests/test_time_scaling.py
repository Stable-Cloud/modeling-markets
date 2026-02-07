import numpy as np
from market.substrate import MarketSubstrate


def simulate(T, dt, seed):
    m = MarketSubstrate(0.0, [100.0], 0.05, 0.2, seed=seed)
    for _ in range(int(T / dt)):
        m.advance_time(dt)
    return np.log(m.prices[0])


def test_variance_scaling():
    dt = 0.01
    N = 10000

    Ts = [1.0, 2.0, 4.0]
    vars_ = []

    for T in Ts:
        samples = [simulate(T, dt, i) for i in range(N)]
        vars_.append(np.var(samples))

    ratio_2 = vars_[1] / vars_[0]
    ratio_4 = vars_[2] / vars_[0]

    assert abs(ratio_2 - 2.0) < 0.2
    assert abs(ratio_4 - 4.0) < 0.4
