# simulation/log.py

class EventLog:
    """
    Records discrete events during the simulation.
    """

    def __init__(self):
        self.records = []

    def record(self, time, card, market_state):
        self.records.append({
            "time": time,
            "card": card,
            "state": market_state
        })

    def __iter__(self):
        return iter(self.records)
