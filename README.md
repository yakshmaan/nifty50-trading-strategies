# Nifty 50 Trading Strategies Backtest

An honest experimental study comparing two popular trading strategies on the Nifty 50 index.

### Project Overview
- **Period**: January 2021 – January 2024
- **Strategies Tested**:
  - 20/50-day Simple Moving Average (SMA) Crossover
  - Random Forest Classifier with 17 technical indicators

### Key Results
- Buy & Hold Return: **55.02%**
- Moving Average Crossover: **12.37%** (17 trades)
- Random Forest: **5.34%** (50 trades)

### Main Findings
The Random Forest model showed severe overfitting (100% training accuracy but only 50.48% on test data). Both strategies significantly underperformed simple buy-and-hold after including realistic transaction costs (0.2% round-trip).

This project demonstrates why basic technical analysis and machine learning strategies built on public data often fail to beat the market in practice.

### Repository Contents
- Full research paper (PDF)
- Python code for data collection, strategies, and backtesting


### Technologies Used
- Python
- yfinance
- pandas, numpy
- scikit-learn (Random Forest)
- matplotlib / seaborn

---

**Note**: This was my first major research project in quantitative finance. The goal was to test real-world performance honestly rather than chasing impressive-looking backtest results.

Feedback and suggestions are welcome!
## About the Author

**Yaksh**  
Independent Researcher | Finance & Data Enthusiast  
mohali, Punjab, India  

I created this project to honestly evaluate whether simple technical analysis and basic machine learning strategies can beat the market. After running the backtests, I learned valuable lessons about overfitting, transaction costs, and the limitations of public data in trading.

This repository contains the full research paper and Python code for both strategies.

Open to feedback, discussions, and collaboration.

📧 your.email@example.com  
🔗 [LinkedIn]www.linkedin.com/in/yaksh1
