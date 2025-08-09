# BackTestEngine

## Overview
`BackTestEngine.py` is a Python module for backtesting quantitative trading strategies on multiple financial securities.  
It uses predictive models to forecast short-term returns, applies configurable trading rules (including transaction fees), and simulates portfolio performance over time.

The engine supports:
- Multiple securities (e.g., cryptocurrencies, stocks)
- Multiple predictive models
- Customizable trading rules
- Fee-aware rebalancing logic
- Portfolio value tracking

---

## Features
- **Model-Agnostic Predictions**  
  Works with any model implementing `.predict()`, including `statsmodels` regression models and machine learning frameworks.

- **Dynamic Trading Rules**  

- **Support for NaN Prices**  
  Handles missing price data by marking assets as non-tradable.

- **Fee Handling**  
  Buy and sell fees in **basis points** (bps) applied during portfolio rebalancing.

- **Simulation Outputs**  
  Produces time series of portfolio values and can plot performance.

---

## File Structure
- [`BackTestEngine`](https://github.com/juventino999/BackTestEngine/blob/main/BackTestEngine.py): Core simulation engine
  - Generates predictions from models.  
  - Computes portfolio weights.  
  - Runs the backtest: executes trades and updates portfolio values.  
- [`Security`](https://github.com/juventino999/BackTestEngine/blob/main/Security.py): Object representing a security, tracking name, shares, value, and tradability.
- [`BackTestHelper`](https://github.com/juventino999/BackTestEngine/blob/main/BackTestHelper.py): Utility class for generating buy/sell/hold recommendations and calculating portfolio weights.


