# Inventory-Adjusted Market Making Simulator

A high-frequency trading backtester comparing inventory-aware strategies against naive fixed-spread approaches. Built to analyze market microstructure dynamics and risk management.

## Project Overview

This simulator compares two market making strategies under realistic market conditions:
- **Strategy A (Naive)**: Fixed bid-ask spread with no inventory management.
- **Strategy B (Inventory-Aware)**: Implements the Avellaneda-Stoikov framework to dynamically skew quotes based on inventory position.

The simulation environment uses **Geometric Brownian Motion (GBM)** for price dynamics and **Poisson processes** for order flow.

## The Math

### Price Dynamics
The true mid-price evolves according to Geometric Brownian Motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

### Inventory Risk (Avellaneda-Stoikov)
The inventory-aware strategy skews quotes to induce mean reversion:

$$\delta_b = \delta_a + \gamma \sigma^2 q$$

Where $q$ is the current inventory and $\gamma$ is the risk aversion parameter.

## Results

Running the simulation demonstrates that inventory management significantly improves risk-adjusted returns:

| Metric | Strategy A (Naive) | Strategy B (Inventory-Aware) |
|--------|-------------------|------------------------------|
| **PnL** | Negative (Loss) | Positive (Profit) |
| **Inventory Volatility** | High | Low (Mean Reverting) |
| **Sharpe Ratio** | Negative | Positive |

**Key Insight:** Strategy A often accumulates toxic inventory during trends, while Strategy B pays a small spread cost to offload risk, leading to long-term survival.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt