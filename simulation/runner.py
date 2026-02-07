# simulation/runner.py

class SimulationRunner:
    """
    Coordinates market evolution, events,
    agent decisions, execution, and portfolio tracking.
    """

    def __init__(
        self,
        market,
        event_engine,
        event_log,
        agents,
        execution_engine,
        portfolio_trackers
    ):
        self.market = market
        self.engine = event_engine
        self.log = event_log
        self.agents = agents
        self.execution_engine = execution_engine
        self.portfolio_trackers = portfolio_trackers
        self.market_history = []

    def run(self, dt, max_events=None):
        count = 0

        while True:
            card = self.engine.step_until_event(dt)
            state = self.market.get_state()
            self.market_history.append(state.copy())
            # Log the event
            self.log.record(
                time=state["time"],
                card=card,
                market_state=state
            )

            # Agents act
            for agent in self.agents:
                action = agent.observe_and_propose(card, state)
                self.execution_engine.execute(agent, action, state)

            # Record portfolios AFTER execution
            for tracker in self.portfolio_trackers:
                tracker.record(state)

            count += 1
            if max_events is not None and count >= max_events:
                break
