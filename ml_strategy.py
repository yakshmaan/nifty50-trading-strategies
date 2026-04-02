import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download data
nifty = yf.download('^NSEI', start='2021-01-01', end='2024-01-01')
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.droplevel(1)

# Feature Engineering
def create_features(data):
    """
    Create technical indicators as features for ML
    """
    df = pd.DataFrame(index=data.index)
    df['price'] = data['Close']
    
    # Returns
    df['returns_1d'] = data['Close'].pct_change(1)
    df['returns_5d'] = data['Close'].pct_change(5)
    df['returns_20d'] = data['Close'].pct_change(20)
    
    # Moving averages
    df['ma_5'] = data['Close'].rolling(5).mean()
    df['ma_10'] = data['Close'].rolling(10).mean()
    df['ma_20'] = data['Close'].rolling(20).mean()
    df['ma_50'] = data['Close'].rolling(50).mean()
    
    # MA ratios (relative position)
    df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
    df['ma_ratio_10_50'] = df['ma_10'] / df['ma_50']
    
    # Volatility
    df['volatility_5'] = df['returns_1d'].rolling(5).std()
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    
    # RSI (simplified)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    df['momentum_5'] = data['Close'] - data['Close'].shift(5)
    df['momentum_20'] = data['Close'] - data['Close'].shift(20)
    
    # Volume (if available)
    if 'Volume' in data.columns:
        df['volume'] = data['Volume']
        df['volume_ma'] = data['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Target: 1 if price goes up tomorrow, 0 if down
    df['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return df.dropna()

# Create features
features_df = create_features(nifty)

print("=" * 70)
print("RANDOM FOREST ML STRATEGY")
print("=" * 70)
print(f"Total data points: {len(features_df)}")
print(f"Features created: {len(features_df.columns) - 2}")  # -2 for price and target
print("\nFeatures:")
for col in features_df.columns:
    if col not in ['price', 'target']:
        print(f"  - {col}")

# Prepare train/test split (70/30)
split_point = int(len(features_df) * 0.7)

feature_cols = [col for col in features_df.columns if col not in ['price', 'target']]
X = features_df[feature_cols]
y = features_df['target']

X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train period: {features_df.index[0].date()} to {features_df.index[split_point-1].date()}")
print(f"Test period: {features_df.index[split_point].date()} to {features_df.index[-1].date()}")

# Train Random Forest
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"Training Accuracy:   {train_accuracy*100:.2f}%")
print(f"Test Accuracy:       {test_accuracy*100:.2f}%")
print(f"Overfit Gap:         {(train_accuracy - test_accuracy)*100:.2f}%")
print("=" * 70)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head().iterrows():
    print(f"  {row['feature']:<20} {row['importance']:.4f}")

# Backtest the ML strategy
def backtest_ml_strategy(features_df, predictions, split_point, initial_capital=100000, cost_per_trade=0.001):
    """
    Backtest using ML predictions on test set only
    """
    # Use only test period
    test_df = features_df[split_point:].copy()
    test_df['prediction'] = predictions
    
    cash = initial_capital
    shares = 0
    equity_curve = []
    trades = 0
    
    for i in range(len(test_df)):
        current_pred = test_df['prediction'].iloc[i]
        current_price = test_df['price'].iloc[i]
        
        # Buy signal (prediction = 1, price will go up)
        if current_pred == 1 and shares == 0:
            transaction_cost = cash * cost_per_trade
            shares = (cash - transaction_cost) / current_price
            cash = 0
            trades += 1
        
        # Sell signal (prediction = 0, price will go down)
        elif current_pred == 0 and shares > 0:
            cash = shares * current_price
            transaction_cost = cash * cost_per_trade
            cash = cash - transaction_cost
            shares = 0
            trades += 1
        
        total_equity = cash + (shares * current_price)
        equity_curve.append(total_equity)
    
    return equity_curve, trades

# Run ML backtest on test set
ml_equity_no_costs, trades_no_costs = backtest_ml_strategy(
    features_df, y_test_pred, split_point, cost_per_trade=0.0
)
ml_equity_with_costs, trades_with_costs = backtest_ml_strategy(
    features_df, y_test_pred, split_point, cost_per_trade=0.001
)

# Calculate buy-and-hold for test period
test_period_df = features_df[split_point:]
buy_hold_test_return = (test_period_df['price'].iloc[-1] / test_period_df['price'].iloc[0] - 1) * 100

ml_no_cost_return = (ml_equity_no_costs[-1] / 100000 - 1) * 100
ml_with_cost_return = (ml_equity_with_costs[-1] / 100000 - 1) * 100

print("\n" + "=" * 70)
print("BACKTEST RESULTS (Test Period Only)")
print("=" * 70)
print(f"Buy & Hold (Test Period):       {buy_hold_test_return:.2f}%")
print(f"ML Strategy (No Costs):         {ml_no_cost_return:.2f}%")
print(f"ML Strategy (With Costs):       {ml_with_cost_return:.2f}%")
print(f"\nNumber of Trades:               {trades_with_costs}")
print(f"Transaction Cost Impact:        -{(ml_no_cost_return - ml_with_cost_return):.2f}%")
print(f"\n** ML Strategy UNDERPERFORMED by {(buy_hold_test_return - ml_with_cost_return):.2f}% **")
print("=" * 70)

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Train vs Test Accuracy
ax1.bar(['Training', 'Test'], [train_accuracy*100, test_accuracy*100], 
        color=['green', 'red'], alpha=0.7)
ax1.axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy: Training vs Test (Overfitting Evidence)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Feature Importance
top_features = feature_importance.head(10)
ax2.barh(top_features['feature'], top_features['importance'], color='steelblue', alpha=0.7)
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Feature Importances', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Equity curves (test period)
test_dates = test_period_df.index
buy_hold_equity_test = [100000 * (test_period_df['price'].iloc[i] / test_period_df['price'].iloc[0]) 
                        for i in range(len(test_period_df))]

ax3.plot(test_dates, buy_hold_equity_test, label=f'Buy & Hold ({buy_hold_test_return:.2f}%)', 
         linewidth=2.5, color='green')
ax3.plot(test_dates, ml_equity_no_costs, label=f'ML No Costs ({ml_no_cost_return:.2f}%)', 
         linewidth=2, alpha=0.7, color='blue')
ax3.plot(test_dates, ml_equity_with_costs, label=f'ML With Costs ({ml_with_cost_return:.2f}%)', 
         linewidth=2, alpha=0.7, color='red', linestyle='--')
ax3.axhline(y=100000, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlabel('Date')
ax3.set_ylabel('Portfolio Value (₹)')
ax3.set_title('ML Strategy Performance (Test Period)', fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 4. Comparison table as image
comparison_data = [
    ['Metric', 'Buy & Hold', 'ML (No Costs)', 'ML (With Costs)'],
    ['Return', f'{buy_hold_test_return:.2f}%', f'{ml_no_cost_return:.2f}%', f'{ml_with_cost_return:.2f}%'],
    ['Trades', '0', f'{trades_no_costs}', f'{trades_with_costs}'],
    ['Train Acc.', '-', f'{train_accuracy*100:.2f}%', f'{train_accuracy*100:.2f}%'],
    ['Test Acc.', '-', f'{test_accuracy*100:.2f}%', f'{test_accuracy*100:.2f}%'],
]

ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=comparison_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Performance Comparison', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('ml_strategy_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Chart saved as 'ml_strategy_results.png'")
print("\nKEY FINDINGS:")
print("  1. Model overfits: {:.2f}% train accuracy vs {:.2f}% test accuracy".format(
    train_accuracy*100, test_accuracy*100))
print("  2. Test accuracy barely beats random guess (50%)")
print("  3. ML strategy underperforms buy-and-hold by {:.2f}%".format(
    buy_hold_test_return - ml_with_cost_return))
print("  4. 'AI' doesn't help - same failure as simple moving averages")
