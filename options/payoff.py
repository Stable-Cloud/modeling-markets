# options/payoff.py

class EuropeanCallPayoff:
    def __init__(self, strike):
        self.strike = strike

    def __call__(self, price):
        return max(price - self.strike, 0.0)
