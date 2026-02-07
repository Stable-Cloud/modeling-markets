from market.substrate import MarketSubstrate


def test_no_jumps():
    m = MarketSubstrate(0.0, [100.0], 0.05, 0.2, seed=42)
    dt = 0.001

    prev_price = m.prices[0]
    for _ in range(1000):
        m.advance_time(dt)
        assert abs(m.prices[0] - prev_price) < 10.0
        prev_price = m.prices[0]

