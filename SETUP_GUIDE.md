# Market Making Simulator (Inventory-Aware)

A high-frequency trading simulation comparing inventory-aware strategies against naive fixed-spread approaches. This project explores market microstructure dynamics, adverse selection, and inventory risk management.

## Project Overview

This simulator compares two market making strategies under stochastic market conditions:
- **Strategy A (Naive)**: Fixed bid-ask spread with no inventory management.
- **Strategy B (Inventory-Aware)**: Implements an inventory skew logic (inspired by Avellaneda-Stoikov) to dynamically adjust quotes based on position.

The simulation environment uses **Geometric Brownian Motion (GBM)** for price dynamics and **Poisson processes** for order flow.

## The Math

### Price Dynamics
The true mid-price evolves according to Geometric Brownian Motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

### Inventory Risk Logic
The inventory-aware strategy skews quotes to encourage mean reversion of inventory:

$$\text{Skew} = -q \cdot \gamma \cdot \sigma^2$$

Where $q$ is current inventory and $\gamma$ is the risk aversion parameter.
- If $q > 0$ (Long), we lower quotes to sell.
- If $q < 0$ (Short), we raise quotes to buy.

## Key Features

### 1. Code Structure
- **Object-Oriented Design**: Clean class hierarchy for strategies.
- **Vectorized Operations**: Efficient NumPy calculations.
- **Type Hints**: Standard Python typing for clarity.

### 2. Quantitative Components
- **GBM Implementation**: Log-normal price paths.
- **Inventory Skewing**: Implementation of risk aversion parameters.
- **Order Flow**: Modeled using Poisson arrival times.

## Results

Running the simulation typically demonstrates:

| Metric | Strategy A (Naive) | Strategy B (Inventory-Aware) |
|--------|-------------------|------------------------------|
| **PnL** | High Variance / Negative | Consistent / Positive |
| **Inventory** | Drifts (High Risk) | Mean Reverting (Low Risk) |
| **Sharpe Ratio** | Low | High |

**Key Insight:** Strategy A often accumulates "toxic" inventory during trends, while Strategy B pays a small spread cost to offload risk, ensuring survival.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
