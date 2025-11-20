# HFT Market Making Simulator

I built this project to understand the mechanics of high-frequency trading, specifically how market makers manage inventory risk. 

Most basic tutorials assume you just set a bid and an ask and collect the spread. In reality, if you don't manage your inventory, you get run over by price trends. This simulator proves that by comparing a "naive" strategy against the Avellaneda-Stoikov framework.

## Project Overview

The simulation runs two strategies on the same price path:

1.  **Strategy A (Naive)**: Quotes a fixed spread around the mid-price. It ignores how much stock it's holding.
2.  **Strategy B (Inventory-Aware)**: Skews its quotes based on current inventory. If it's long, it lowers quotes to sell. If it's short, it raises quotes to buy.

I used Python for the simulation logic and modeled the market dynamics using stochastic processes (GBM and Poisson).

## The Math Behind It

To make the simulation realistic, I couldn't just use random walks. I implemented the following models:

### 1. Price Dynamics
The true price follows Geometric Brownian Motion (GBM). I used this because it stops prices from going negative and captures volatility better than a simple random walk.

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

### 2. Inventory Skew (Avellaneda-Stoikov)
This is the core logic for Strategy B. The mid-price quote is adjusted based on the inventory position $q$:

$$\text{Quote}_{\text{mid}} = \text{True}_{\text{mid}} - q \cdot \gamma \cdot \sigma^2$$

* $q$: Current inventory (positive = long, negative = short)
* $\gamma$: Risk aversion parameter
* $\sigma^2$: Volatility

Basically, as inventory ($q$) gets larger, the skew gets bigger, forcing the system to dump the position faster.

### 3. Order Flow
Market orders arrive via a Poisson process. I set it up so that the probability of a trade happening actually depends on how close my quote is to the fair price. This rewards Strategy B for skewing its quotes correctly.

## Results

Running the simulation for 10,000 ticks showed a massive difference in performance.

| Metric | Strategy A (Naive) | Strategy B (Inventory-Aware) |
|--------|-------------------|------------------------------|
| **PnL** | -$22,322 (Loss) | +$7,336 (Profit) |
| **Max Inventory** | ±100 (Hit limits constantly) | ±100 (Rarely hit limits) |
| **Avg Inventory** | ~47 shares | ~36 shares |

**Key Takeaway**: Strategy A went bankrupt because it held onto losing positions during trends. Strategy B took small hits on the spread to offload risk, which ended up being profitable in the long run.

## How to Run

You need Python 3.8+ and the libraries in `requirements.txt`.

1.  Clone the repo
    ```bash
    git clone [https://github.com/Ronak-Mahajan/hft-market-maker.git](https://github.com/Ronak-Mahajan/hft-market-maker.git)
    ```

2.  Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the notebook
    ```bash
    jupyter notebook simulation.ipynb
    ```

## References

* Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book*.
* Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk*.

---

*Created by Ronak Mahajan*