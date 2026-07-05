import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 3: Backtesting & Performance Analysis
In this notebook, we simulate our trading strategy over the 2-year testing period to see if it survives transaction costs.

**Constraints:**
- Initial Capital: $1,000,000
- Transaction Costs: 10 bps (0.10%) per trade
- Allocation: Equal-Weight across all active Long and Short signals for the day.
"""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/signals_v2.csv'
"""

md_loading = """## 1. Load Data and Pivot
We load the `signals.csv` file and pivot the prices and signals into a matrix format (Rows = Dates, Columns = Assets)."""

code_loading = """df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

# Pivot prices and signals
prices = df.pivot(index='Date', columns='Asset', values='Close')
signals = df.pivot(index='Date', columns='Asset', values='Signal').fillna(0)

# Calculate daily asset returns
# Note: if you buy at close of day t-1, your return on day t is (Close_t / Close_{t-1}) - 1
asset_returns = prices.pct_change().shift(-1) # Shift -1 so row T aligns with the return of T to T+1
"""

md_portfolio = """## 2. Portfolio Construction & Transaction Costs
We allocate our portfolio equally among all active signals (both Long and Short).
We also calculate portfolio turnover to accurately deduct the 10 bps transaction cost."""

code_portfolio = """# 1. Calculate Target Weights
# Count the number of active signals each day
active_signals_count = signals.abs().sum(axis=1)

# Avoid division by zero on days with no signals
weights = signals.div(active_signals_count.replace(0, 1), axis=0)

# 2. Calculate Gross Daily Returns of the Strategy
# Strategy return = Sum of (Weight in asset i * Return of asset i)
gross_strategy_returns = (weights * asset_returns).sum(axis=1)

# 3. Calculate Turnover and Transaction Costs
# Turnover is the absolute change in weights from one day to the next
turnover = weights.diff().abs().sum(axis=1)
turnover.fillna(0, inplace=True)

# Transaction cost = 10 bps (0.0010) per unit of turnover
tc_rate = 0.0010
transaction_costs = turnover * tc_rate

# 4. Net Strategy Returns
net_strategy_returns = gross_strategy_returns - transaction_costs

# Combine for comparison
portfolio = pd.DataFrame({
    'Gross_Return': gross_strategy_returns,
    'Net_Return': net_strategy_returns,
    'Turnover': turnover
})

# Drop the last row since return is NaN (shifted)
portfolio.dropna(inplace=True)
"""

md_metrics = """## 3. Performance Metrics
We calculate Sharpe Ratio, Maximum Drawdown, and Total Return."""

code_metrics = """def calc_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

metrics = []
for col in ['Gross_Return', 'Net_Return']:
    ret = portfolio[col]
    total_return = (1 + ret).prod() - 1
    # Annualized Sharpe (assuming 252 trading days)
    sharpe = np.sqrt(252) * (ret.mean() / ret.std())
    max_dd = calc_drawdown(ret)
    
    metrics.append({
        'Strategy': col,
        'Total Return (%)': total_return * 100,
        'Annualized Sharpe': sharpe,
        'Max Drawdown (%)': max_dd * 100
    })

metrics_df = pd.DataFrame(metrics)
print("--- PERFORMANCE METRICS ---")
display(metrics_df)

print(f"\\nAverage Daily Turnover: {portfolio['Turnover'].mean():.2%}")
"""

md_plot = """## 4. Cumulative PnL Visualization
We plot the cumulative equity curve of $1,000,000 starting capital."""

code_plot = """initial_capital = 1000000

# Calculate equity curves
equity_gross = initial_capital * (1 + portfolio['Gross_Return']).cumprod()
equity_net = initial_capital * (1 + portfolio['Net_Return']).cumprod()

# Benchmark: Equal weight buy and hold all assets
# Just average the returns of all assets each day
benchmark_returns = asset_returns.mean(axis=1)
benchmark_returns.dropna(inplace=True)
equity_benchmark = initial_capital * (1 + benchmark_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(equity_gross, label='Strategy (Gross)', linestyle='--')
plt.plot(equity_net, label='Strategy (Net after 10bps TC)', linewidth=2)
plt.plot(equity_benchmark, label='Benchmark (Equal Weight Buy & Hold)', alpha=0.7)

plt.title('Strategy Performance vs Benchmark ($1,000,000 Initial Capital)')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_header),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(md_loading),
    nbf.v4.new_code_cell(code_loading),
    nbf.v4.new_markdown_cell(md_portfolio),
    nbf.v4.new_code_cell(code_portfolio),
    nbf.v4.new_markdown_cell(md_metrics),
    nbf.v4.new_code_cell(code_metrics),
    nbf.v4.new_markdown_cell(md_plot),
    nbf.v4.new_code_cell(code_plot)
]

with open('Part3_Backtesting.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook Part3_Backtesting.ipynb generated successfully!")
