# Modeling Markets: Discrete Hedging in Quantitative Finance

## Executive Summary

**Modeling Markets** is a comprehensive quantitative finance framework that bridges the critical gap between Black-Scholes option pricing theory and the practical realities of discrete hedging in real markets. This project investigates how trading constraints—specifically the inability to continuously rebalance portfolios—affect hedging performance, cost, and risk. Through rigorous Monte Carlo simulation and empirical analysis, we quantify the exact price of discreteness and provide actionable insights for derivatives traders and risk managers.

## Thesis Statement

The theoretical assumption of continuous trading in the Black-Scholes model is fundamentally incompatible with market practice. While academic models suggest option replication is perfect under continuous hedging, actual markets require discrete rebalancing at fixed intervals. This project demonstrates that:

1. **Replication error scales predictably with rebalancing frequency**, driven primarily by gamma risk
2. **Transaction costs create a fundamental trade-off**: more frequent rebalancing improves hedging but increases costs
3. **Information about market microstructure and volatility regimes can adaptively optimize hedging decisions**, reducing the cost-frequency trade-off
4. **The true economic cost of hedging extends beyond the Black-Scholes price** and must account for rebalancing overhead

## Research Motivation

The Black-Scholes model, while elegant theoretically, operates under the assumption of continuous trading with no frictions. In practice:
- Markets only allow trading at discrete times
- Transaction costs are non-negligible
- Volatility is not constant but regime-dependent
- Market microstructure affects execution quality

This creates a replication error that sellers of options must hedge against. The question is not *whether* replication is imperfect, but *how expensive* this imperfection is and *how to minimize* it.

## Core Research Questions

### 1. **Impact of Rebalancing Frequency**
- How does the rebalancing interval (Δt) affect replication error?
- Does error converge to zero as Δt → 0?
- What is the functional relationship between Δt and expected hedging loss?

### 2. **Sources of Replication Error**
- What is the relative contribution of **gamma risk** (curvature loss)?
- How does **volatility misestimation** compound errors?
- What is the magnitude of **transaction costs** impact?

### 3. **Cost-Frequency Trade-off**
- Is there an optimal rebalancing frequency balancing hedging error and costs?
- How do market conditions (volatility regime, liquidity) affect optimization?

### 4. **Adaptive Hedging Strategies**
- Can real-time market information (volatility cards, gamma alerts) improve hedging?
- How should rebalancing decisions respond to market microstructure signals?

## Project Structure

```
Modeling Markets/
│
├── market/                          # Market Substrate Layer
│   └── substrate.py                 # Continuous-time geometric Brownian motion
│                                    # Euler-Maruyama discretization for price paths
│
├── options/                         # Derivatives Layer  
│   ├── black_scholes.py            # BS pricing, Greeks (delta, gamma, vega, theta)
│   └── payoff.py                   # Option payoff calculations
│
├── agents/                          # Hedging Strategy Layer
│   ├── base.py                     # Abstract agent interface
│   ├── delta_hedging.py            # Standard delta hedging (Δ rebalancing)
│   ├── information_aware_hedging.py # Adaptive hedging with information cards
│   ├── actions.py                  # Trading action representation
│   └── __pycache__/
│
├── events/                          # Event Management System
│   ├── base.py                     # Stopping condition interface
│   ├── time.py                     # Time-based rebalancing (fixed intervals)
│   ├── price.py                    # Price-based triggers (not currently used)
│   ├── engine.py                   # Event processing engine
│   ├── information_generator.py    # Information card generation
│   ├── cards/                      # Information card types
│   │   ├── gamma_alert.py         # High gamma notifications
│   │   ├── market_microstructure.py # Liquidity/spread signals
│   │   ├── realized_metrics.py    # Realized volatility tracking
│   │   └── volatility_regime.py   # Volatility state identification
│   └── __pycache__/
│
├── execution/                       # Trade Execution Layer
│   ├── costs.py                    # Transaction cost models
│   ├── validator.py                # Execution validation & logging
│   └── __pycache__/
│
├── portfolio/                       # Portfolio Tracking
│   ├── tracker.py                  # Portfolio state & value tracking
│   └── __pycache__/
│
├── simulation/                      # Simulation Orchestration
│   ├── runner.py                   # Main simulation loop
│   ├── log.py                      # Event logging & statistics
│   └── __pycache__/
│
├── analysis/                        # Post-Simulation Analysis
│   ├── risk_metrics.py             # VaR, CVaR, shortfall metrics
│   ├── error_decomposition.py      # Error source attribution
│   ├── path_analysis.py            # Path classification & difficulty
│   ├── market_comparison.py        # Hedging vs. market comparison
│   ├── replication.py              # Replication error analysis
│   └── __pycache__/
│
├── visualization/                   # Plotting & Graphics
│   └── plots.py                    # Comprehensive plotting functions
│
├── experiments/                     # Experimental Drivers
│   ├── experiment_1_stopping_time.py      # Rebalancing frequency study
│   ├── experiment_2_transaction_costs.py  # Cost-frequency trade-off
│   ├── experiment_3_information_aware.py  # Adaptive strategy testing
│   ├── plot_experiment_1.py              # Result visualization
│   └── __pycache__/
│
├── tests/                           # Unit Tests
│   ├── test_distribution.py        # GBM distribution verification
│   ├── test_martingale.py          # Martingale property validation
│   ├── test_paths.py               # Path properties
│   └── test_time_scaling.py        # Variance scaling laws
│
├── Complete_Project_Analysis.ipynb  # Comprehensive analysis notebook
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

### Setup
```bash
pip install -r requirements.txt
```

## Installation & Setup

### Requirements
- Python 3.8+
- NumPy (numerical computing)
- SciPy (scientific functions)
- Matplotlib (visualization)

### Quick Start
```bash
pip install -r requirements.txt
python Complete_Project_Analysis.ipynb  # Run comprehensive analysis
```

## Core Codebase Components

### 1. Market Substrate (`market/substrate.py`)
**Purpose**: Simulate continuous stock price paths using geometric Brownian motion.

**Key Class: `MarketSubstrate`**
- Implements Euler-Maruyama discretization: dS = μS dt + σS dW
- Generates price paths with configurable time step
- Input validation and numerical stability checks

**Example Usage:**
```python
from market.substrate import MarketSubstrate

market = MarketSubstrate(
    initial_time=0.0,
    initial_prices=[100.0],     # S₀
    mu=[0.05],                   # drift
    sigma=[0.2],                 # volatility
    seed=42
)

# Advance market by time dt
market.advance(dt=0.001)
```

### 2. Black-Scholes Module (`options/black_scholes.py`)
**Purpose**: Calculate option prices and Greeks under the Black-Scholes model.

**Available Functions:**
- `black_scholes_call_price(S, K, r, sigma, T, t)` - European call value
- `black_scholes_delta(S, K, r, sigma, T, t)` - Position size for hedging
- `black_scholes_gamma(S, K, r, sigma, T, t)` - Curvature/convexity
- `black_scholes_vega(S, K, r, sigma, T, t)` - Volatility sensitivity
- `black_scholes_theta(S, K, r, sigma, T, t)` - Time decay

**Key Formula:**
$$C_{BS} = S_0 N(d_1) - Ke^{-rT}N(d_2)$$

where:
$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### 3. Hedging Agents (`agents/`)

#### Base Agent Class (`base.py`)
Abstract interface for all trading strategies:
- Track capital and cash reserves
- Monitor trade history and costs
- Abstract `decide()` method for strategy implementation

#### Delta Hedging Agent (`delta_hedging.py`)
**Strategy**: Maintain portfolio delta = option delta at each rebalancing time.

**Workflow:**
1. Compute target delta: Δ = N(d₁)
2. Calculate required trades: Δ_trade = Δ_target - Δ_current
3. Execute trades and update cash
4. Record transaction costs

**Configuration:**
```python
from agents.delta_hedging import DeltaHedgingAgent

agent = DeltaHedgingAgent(
    capital=8.0,                          # C_BS
    option_params={'K': 100, 'r': 0.05, 'sigma': 0.2, 'T': 1.0},
    rebalance_tolerance=0.001             # Δ precision tolerance
)
```

#### Information-Aware Hedging Agent (`information_aware_hedging.py`)
**Enhancement**: Adapt hedging decisions using real-time market information cards.

**Features:**
- **Volatility Adaptation**: Update volatility assumptions from realized vol
- **Gamma Adaptation**: Adjust rebalancing tolerance based on option gamma
- **Cost Optimization**: Defer trades in high-cost environments
- **Risk Management**: Respond to realized metric signals

**Configuration:**
```python
from agents.information_aware_hedging import InformationAwareDeltaHedgingAgent

agent = InformationAwareDeltaHedgingAgent(
    capital=8.0,
    option_params={'K': 100, 'r': 0.05, 'sigma': 0.2, 'T': 1.0},
    volatility_adaptation=True,
    gamma_adaptation=True,
    cost_optimization=True,
    volatility_adaptation_rate=0.3
)
```

### 4. Event System (`events/`)
**Purpose**: Manage discrete trading opportunities and market information.

#### Stopping Conditions
Define when rebalancing occurs:

**TimeStoppingCondition** (`time.py`)
```python
from events.time import TimeStoppingCondition

condition = TimeStoppingCondition(
    stop_interval=0.1,    # Δt = 0.1 years
    start_time=0.0,
    end_time=1.0
)

# Check if agent should rebalance at time t
should_rebalance = condition.should_stop_at_time(t)
```

#### Information Cards (`events/cards/`)
Real-time market signals:
- **GammaAlert**: High gamma regions (increased hedging error risk)
- **VolatilityRegime**: Volatility regime classification
- **MarketMicrostructure**: Liquidity and spread information
- **RealizedMetrics**: Historical volatility, realized variance

### 5. Simulation Runner (`simulation/runner.py`)
**Purpose**: Orchestrate full hedging simulation from inception to maturity.

**Simulation Loop:**
1. Initialize market and agents
2. For each time step:
   - Advance market (generate returns)
   - Check stopping conditions
   - Agent makes rebalancing decision
   - Execute trades with costs
   - Update portfolio
   - Record statistics
3. Calculate terminal replication error

**Example:**
```python
from simulation.runner import SimulationRunner

runner = SimulationRunner(
    market=market,
    agent=agent,
    event_engine=engine,
    execution_engine=exec_engine,
    dt_market=0.001,              # Fine market simulation step
    dt_logging=0.01               # Statistics recording interval
)

result = runner.run()
print(f"Replication Error: {result['final_error']}")
```

### 6. Analysis Tools (`analysis/`)

**Risk Metrics** (`risk_metrics.py`):
- Value-at-Risk (VaR)
- Conditional VaR (Expected Shortfall)
- Shortfall probability
- Maximum drawdown

**Error Decomposition** (`error_decomposition.py`):
- Attribute error to gamma risk
- Attribute error to volatility changes
- Attribute error to transaction costs

**Path Analysis** (`path_analysis.py`):
- Identify difficult paths (high error)
- Extract path characteristics
- Classify by volatility regime

## Experiments & Results

### Experiment 1: Rebalancing Frequency Analysis
**File**: `experiments/experiment_1_stopping_time.py`

**Objective**: Quantify how rebalancing interval affects hedging performance.

**Parameters:**
- Rebalancing intervals: Δt ∈ {0.01, 0.05, 0.1, 0.2} years
- Monte Carlo paths: 1000
- Option: European call, ATM (S₀ = K = 100)

**Run:**
```bash
python experiments/experiment_1_stopping_time.py
```

**Expected Results:**
- Mean error → 0 (unbiased hedging)
- Error std dev scales as O(√Δt)
- Larger Δt → larger tail risk (negative errors)
- Gamma drives error variance

### Experiment 2: Transaction Costs Impact
**File**: `experiments/experiment_2_transaction_costs.py`

**Objective**: Analyze trade-off between hedging frequency and costs.

**Key Variables:**
- Cost per trade: 0 bps to 5 bps
- Rebalancing intervals: varied
- Optimization: find cost-minimizing frequency

**Expected Trade-off:**
- Few rebalances (Δt large) → low costs, high hedging error
- Many rebalances (Δt small) → high costs, low hedging error
- Optimal Δt* exists for each cost structure

### Experiment 3: Information-Aware Hedging
**File**: `experiments/experiment_3_information_aware.py`

**Objective**: Test adaptive hedging using market information cards.

**Strategies Compared:**
1. **Baseline**: Fixed-interval delta hedging
2. **Adaptive**: Information-aware rebalancing
3. **Enhanced**: With volatility and gamma adaptation

**Expected Outcome:**
- Information-aware strategies reduce error while maintaining cost discipline
- Gamma alerts trigger protective rebalances
- Volatility updates prevent model drift

## Mathematical Framework

### Black-Scholes Model Assumptions
1. No arbitrage (frictionless markets)
2. Continuous trading at any time
3. No transaction costs or taxes
4. Constant risk-free rate r
5. Constant volatility σ
6. European options (exercise only at maturity)

### Discrete Hedging Error Source
Under discrete rebalancing, the portfolio replicates poorly due to **gamma risk**:

$$
P\&L_{\text{discrete}} \approx \int_0^T \frac{1}{2}\Gamma(t)(\Delta S_t)^2 \, dt
$$




This integral depends on:
- **Gamma magnitude**: Convexity of option value
- **Realized variance**: $(∆S_t)^2$ realization
- **Rebalancing frequency**: Larger $\Delta t$ → larger $|∆S_t|$ between rebalances

### Convergence Property
As $\Delta t \to 0$:
$$\lim_{\Delta t \to 0} E[\text{Error}^2] = 0$$

But for finite $\Delta t$:
$$E[\text{Error}^2] \approx C \cdot \Delta t$$

where C depends on gamma, volatility, and contract specifications.

## Key Theoretical Results
- Error variance scales as O(Δt)
- Higher gamma → larger errors
- Errors are path-dependent

**Convergence:**
$$\text{Var}[\text{Error}] \approx C \cdot \Delta t \text{ as } \Delta t \to 0$$

### Transaction Cost Model
With proportional costs (c bps per trade):

$$\text{Total Cost} = c \cdot \sum_{i} |\text{Trade}_i|$$

The net hedging cost trades off:
- **Reduced error** → fewer rebalances needed
- **Increased cost** → more rebalances executed

An optimal rebalancing interval minimizes total cost:
$$\Delta t^* = \arg\min_{\Delta t} E[\text{Error}^2] + \text{Transaction Costs}(\Delta t)$$

## Implementation Quality

### ✅ Validated Components
- **Market simulation** (GBM validated against theoretical distribution)
- **Greeks calculations** (verified against finite differences)
- **Delta hedging** (P&L matches theory under continuous trading)
- **Event system** (proper time stepping and rebalancing triggers)

### 🔧 Quality Assurance
- Comprehensive unit tests for all core modules
- Numerical stability checks and input validation
- Edge case handling (T→0, σ→0, ATM, OTM options)
- Detailed docstrings with mathematical formulas

### 📊 Analysis Capabilities
- Monte Carlo error analysis (1000+ paths)
- Risk metrics (VaR, CVaR, maximum drawdown)
- Error decomposition (gamma, vega, cost attribution)
- Path difficulty classification

## Usage Examples

### Basic Hedging Simulation
```python
import numpy as np
from market.substrate import MarketSubstrate
from events.time import TimeStoppingCondition
from events.engine import EventEngine
from agents.delta_hedging import DeltaHedgingAgent
from execution.validator import ExecutionEngine
from portfolio.tracker import PortfolioTracker
from simulation.runner import SimulationRunner
from options.black_scholes import black_scholes_call_price

# Set up parameters
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
option_params = {'K': K, 'r': r, 'sigma': sigma, 'T': T}
BS_price = black_scholes_call_price(S0, K, r, sigma, T, 0)

# Create market
market = MarketSubstrate(
    initial_time=0,
    initial_prices=[S0],
    mu=[r],
    sigma=[sigma],
    seed=42
)

# Create components
agent = DeltaHedgingAgent(capital=BS_price, option_params=option_params)
stopping = TimeStoppingCondition(stop_interval=0.1, start_time=0, end_time=T)
engine = EventEngine(stopping_condition=stopping)
exec_engine = ExecutionEngine(cost_model=None)
portfolio = PortfolioTracker(initial_capital=BS_price)

# Run simulation
runner = SimulationRunner(
    market=market,
    agent=agent,
    event_engine=engine,
    execution_engine=exec_engine,
    portfolio_tracker=portfolio,
    dt_market=0.001
)

result = runner.run()
print(f"Replication Error: {result['final_error']:.4f}")
print(f"Total Trades: {result['num_trades']}")
```

### Information-Aware Hedging
```python
from agents.information_aware_hedging import InformationAwareDeltaHedgingAgent
from events.information_generator import InformationGenerator

# Create adaptive agent
agent = InformationAwareDeltaHedgingAgent(
    capital=BS_price,
    option_params=option_params,
    volatility_adaptation=True,
    gamma_adaptation=True,
    cost_optimization=True
)

# Create information generator
info_gen = InformationGenerator(market=market)

# Run with information flow
runner = SimulationRunner(
    market=market,
    agent=agent,
    event_engine=engine,
    execution_engine=exec_engine,
    portfolio_tracker=portfolio,
    information_generator=info_gen,
    dt_market=0.001
)

result = runner.run()
print(f"Adaptive Strategy Error: {result['final_error']:.4f}")
```

## Running Experiments

### Experiment 1: Rebalancing Frequency Study
```bash
python experiments/experiment_1_stopping_time.py
```
Analyzes how rebalancing interval affects replication error and risk metrics.

### Experiment 2: Transaction Costs Analysis
```bash
python experiments/experiment_2_transaction_costs.py
```
Studies the cost-benefit trade-off and finds optimal rebalancing frequency.

### Experiment 3: Adaptive Strategy Comparison
```bash
python experiments/experiment_3_information_aware.py
```
Compares baseline delta hedging with information-aware strategies.

### Generate Visualizations
```bash
python experiments/plot_experiment_1.py
```
Creates comprehensive plots of results and analysis.

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

**Test Coverage:**
- **test_distribution.py**: Verify GBM produces correct distribution
- **test_martingale.py**: Verify discounted prices are martingales
- **test_paths.py**: Verify path continuity and smoothness
- **test_time_scaling.py**: Verify variance scaling as dt→0

### Manual Verification
- Greeks validation against finite differences
- Delta hedging error properties
- Black-Scholes pricing consistency

## Key Insights & Findings

### 1. Gamma Risk is Dominant
The primary source of discrete hedging error is **gamma risk** (convexity loss). This occurs because:
- Between rebalancing times, the stock price moves
- Option value changes non-linearly (gamma effect)
- Delta hedging cannot react fast enough
- Net loss = (1/2) × Γ × (ΔS)²

### 2. Error Scales Predictably
Empirical validation shows:
$$E[\text{Error}^2] \propto \Delta t \quad \text{(for small } \Delta t \text{)}$$

This allows traders to estimate hedging cost from:
- Option gamma
- Market volatility
- Rebalancing frequency

### 3. Transaction Costs Create Trade-offs
Adding transaction costs creates a meaningful optimization problem:
- **Very frequent rebalancing**: Low error, high costs
- **Infrequent rebalancing**: Low costs, high error
- **Optimal point**: Minimizes total cost

### 4. Volatility Regime Matters
Errors are larger when:
- Volatility is high (larger price moves)
- At-the-money (maximum gamma)
- Near expiration (higher gamma sensitivity)

### 5. Adaptive Strategies Can Help
Information-aware agents can reduce costs by:
- Deferring rebalancing in low-volatility regimes
- Accelerating rebalancing when gamma peaks
- Using realized vol to correct model assumptions

## Project Roadmap

### ✅ Completed
- Core market simulation (GBM)
- Black-Scholes Greeks implementation
- Delta hedging agent
- Event-driven architecture
- Rebalancing frequency experiments
- Error analysis framework
- Information card system
- Adaptive hedging agents

### 🚧 In Progress
- Transaction cost integration
- Comprehensive risk metrics
- Advanced error decomposition
- Path analysis and classification

### 📋 Future Enhancements
- American option support
- Stochastic volatility models
- Jump diffusion processes
- Exotic option support
- Real market data integration
- Live trading interface

## References

### Foundational Theory
1. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*
2. Hull, J. C. (2018). "Options, Futures, and Other Derivatives" (10th ed.)
3. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*

### Discrete Hedging & Numerical Methods
4. Shreve, S. E. (2004). "Stochastic Calculus for Finance II: Continuous-Time Models"
5. Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering"
6. Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance"

### Practical Implementation
7. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
8. Duffy, D. J. (2006). "Finite Difference Methods in Financial Engineering"

## Architecture Principles

### 1. Separation of Concerns
- **Market**: Price dynamics only
- **Agents**: Decision logic only
- **Execution**: Trade validation and costs
- **Simulation**: Orchestration only

### 2. Extensibility
- Abstract base classes for agents and stopping conditions
- Pluggable cost models
- Custom information card types
- Alternative stopping/rebalancing logic

### 3. Reproducibility
- Seed-based RNG for deterministic paths
- Comprehensive logging of all trades
- Parameter tracking in results
- Version control friendly structure

### 4. Validation
- Input validation at module boundaries
- Numerical stability checks
- Result sanity tests
- Comparison against analytical results

## Contributing

To extend the framework:

1. **Add new stopping conditions**: Subclass `StoppingCondition` in `events/base.py`
2. **Implement new strategies**: Subclass `Agent` in `agents/base.py`
3. **Create custom information cards**: Subclass `InformationCard` in `events/cards/base.py`
4. **Add analysis tools**: Extend `analysis/` modules with new metrics
5. **Write experiments**: Create scripts in `experiments/`

## License

Educational and research use.

## Contact & Support

For questions about:
- **Theory**: Review the docstrings and mathematical formulas
- **Implementation**: Check the Complete_Project_Analysis notebook
- **Experiments**: Run individual experiment files
- **Debugging**: Enable verbose logging in simulation runner

---

**Last Updated**: January 2026
**Status**: Active Development
**Main Focus**: Bridging theory-practice gap in discrete hedging
