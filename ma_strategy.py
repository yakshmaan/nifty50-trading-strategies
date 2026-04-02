import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download data
nifty = yf.download('^NSEI', start='2021-01-01', end='2024-01-01')

# Flatten MultiIndex
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.droplevel(1)

# Moving Average Strategy
def moving_average_strategy(data, short_window=20, long_window=50):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0.0
    signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 1.0
    signals['position_change'] = signals['signal'].diff()
    return signals

# Backtest engine (works for both with/without costs)
def backtest_strategy(signals, initial_capital=100000, cost_per_trade=0.0):
    """
    Universal backtest engine
    cost_per_trade: 0.0 for no costs, 0.001 for 0.1% costs
    """
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['signal'] = signals['signal']
    portfolio['price'] = signals['price']
    
    cash = initial_capital
    shares = 0
    equity_curve = []
    trades = 0
    
    for i in range(len(portfolio)):
        current_signal = portfolio['signal'].iloc[i]
        current_price = portfolio['price'].iloc[i]
        
        if pd.isna(current_signal) or pd.isna(current_price):
            equity_curve.append(cash)
            continue
        
        # Buy signal
        if current_signal == 1.0 and shares == 0:
            transaction_cost = cash * cost_per_trade
            shares = (cash - transaction_cost) / current_price
            cash = 0
            trades += 1
        
        # Sell signal
        elif current_signal == 0.0 and shares > 0:
            cash = shares * current_price
            transaction_cost = cash * cost_per_trade
            cash = cash - transaction_cost
            shares = 0
            trades += 1
        
        total_equity = cash + (shares * current_price)
        equity_curve.append(total_equity)
    
    portfolio['equity'] = equity_curve
    portfolio['trades'] = trades
    return portfolio

# Run strategy
signals = moving_average_strategy(nifty)

# Run backtests
portfolio_no_costs = backtest_strategy(signals, cost_per_trade=0.0)
portfolio_with_costs = backtest_strategy(signals, cost_per_trade=0.001)

# Calculate metrics
no_cost_final = portfolio_no_costs['equity'].iloc[-1]
with_cost_final = portfolio_with_costs['equity'].iloc[-1]
buy_hold_final = 100000 * 1.5502

no_cost_return = (no_cost_final / 100000 - 1) * 100
with_cost_return = (with_cost_final / 100000 - 1) * 100
buy_hold_return = 55.02

num_trades = portfolio_with_costs['trades'].iloc[-1]
cost_impact = no_cost_return - with_cost_return

print("=" * 70)
print("MOVING AVERAGE STRATEGY RESULTS (20/50 MA)")
print("=" * 70)
print(f"Buy & Hold Return:              {buy_hold_return:.2f}%")
print(f"MA Strategy (No Costs):         {no_cost_return:.2f}%")
print(f"MA Strategy (With Costs):       {with_cost_return:.2f}%")
print(f"\nNumber of Trades:               {int(num_trades)}")
print(f"Transaction Cost Impact:        -{cost_impact:.2f}%")
print(f"\n** MA Strategy UNDERPERFORMED by {(buy_hold_return - with_cost_return):.2f}% **")
print("=" * 70)

# Calculate additional metrics
def calculate_metrics(equity_curve, initial_capital=100000):
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    # Sharpe Ratio (assume 6% annual risk-free rate)
    excess_returns = returns - (0.06 / 252)
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Max Drawdown
    cumulative = pd.Series(equity_curve)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return sharpe, max_drawdown

buy_hold_equity = [100000 * (signals['price'].iloc[i] / signals['price'].iloc[0]) 
                   for i in range(len(signals))]

sharpe_bh, dd_bh = calculate_metrics(buy_hold_equity)
sharpe_ma_no, dd_ma_no = calculate_metrics(portfolio_no_costs['equity'])
sharpe_ma_with, dd_ma_with = calculate_metrics(portfolio_with_costs['equity'])

print("\nADVANCED METRICS:")
print("-" * 70)
print(f"{'Strategy':<25} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
print("-" * 70)
print(f"{'Buy & Hold':<25} {sharpe_bh:>13.2f} {dd_bh:>13.2f}%")
print(f"{'MA (No Costs)':<25} {sharpe_ma_no:>13.2f} {dd_ma_no:>13.2f}%")
print(f"{'MA (With Costs)':<25} {sharpe_ma_with:>13.2f} {dd_ma_with:>13.2f}%")
print("=" * 70)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Price + MAs
ax1.plot(signals.index, signals['price'], label='Nifty 50', linewidth=2, color='black')
ax1.plot(signals.index, signals['short_ma'], label='20-day MA', alpha=0.7, color='blue')
ax1.plot(signals.index, signals['long_ma'], label='50-day MA', alpha=0.7, color='red')

# Mark buy/sell signals
buys = signals[signals['position_change'] == 1]
sells = signals[signals['position_change'] == -1]
ax1.scatter(buys.index, buys['price'], color='green', marker='^', s=100, label='Buy', zorder=5)
ax1.scatter(sells.index, sells['price'], color='red', marker='v', s=100, label='Sell', zorder=5)

ax1.set_title('Nifty 50 with Moving Averages & Trade Signals', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (₹)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Equity curves
ax2.plot(signals.index, buy_hold_equity, label='Buy & Hold (55.02%)', 
         linewidth=2.5, color='green')
ax2.plot(portfolio_no_costs.index, portfolio_no_costs['equity'], 
         label=f'MA Strategy - No Costs ({no_cost_return:.2f}%)', 
         linewidth=2, alpha=0.7, color='blue')
ax2.plot(portfolio_with_costs.index, portfolio_with_costs['equity'], 
         label=f'MA Strategy - With Costs ({with_cost_return:.2f}%)', 
         linewidth=2, alpha=0.7, color='red', linestyle='--')
ax2.axhline(y=100000, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')

ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Portfolio Value (₹)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ma_strategy_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Chart saved as 'ma_strategy_results.png'")
print("\nKEY FINDING: Transaction costs reduced returns by {:.2f}%".format(cost_impact))
print("             MA strategy with costs underperformed buy-and-hold by {:.2f}%".format(
    buy_hold_return - with_cost_return))
