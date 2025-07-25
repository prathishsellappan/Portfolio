import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set_palette("husl")

def download_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def calculate_portfolio_metrics(prices, weights, risk_free_rate=0.02):
    returns = prices.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    avg_return = portfolio_returns.mean() * 252
    cov_matrix = returns.cov() * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (avg_return - risk_free_rate) / portfolio_vol
    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252) * -1
    return {
        'avg_return': avg_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe_ratio,
        'daily_returns': portfolio_returns,
        'var_95': var_95,
        'returns_df': returns
    }

def monte_carlo_optimization(prices, num_portfolios=10000, risk_free_rate=0.02):
    returns = prices.pct_change().dropna()
    num_assets = len(prices.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_returns = (returns * weights).sum(axis=1)
        results[0, i] = portfolio_returns.mean() * 252
        results[1, i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]
    
    max_sharpe_idx = np.argmax(results[2])
    return results, weights_record[max_sharpe_idx]

def plot_portfolio_analysis(prices, metrics, weights, tickers, monte_carlo_results):
    fig = plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    for ticker in tickers:
        plt.plot(prices[ticker], label=ticker)
    plt.title('Stock Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(metrics['daily_returns'], color='purple')
    plt.title('Portfolio Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    
    plt.subplot(3, 2, 3)
    plt.pie(weights, labels=tickers, autopct='%1.1f%%')
    plt.title('Optimal Portfolio Allocation')
    
    plt.subplot(3, 2, 4)
    plt.scatter(metrics['volatility'], metrics['avg_return'], s=200, color='red', label='Optimal Portfolio')
    plt.scatter(monte_carlo_results[1, :], monte_carlo_results[0, :], c=monte_carlo_results[2, :], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Annualized)')
    plt.ylabel('Average Return (Annualized)')
    plt.legend()
    
    plt.subplot(3, 2, 5)
    sns.heatmap(metrics['returns_df'].corr(), annot=True, cmap='coolwarm')
    plt.title('Stock Correlation Matrix')
    
    plt.subplot(3, 2, 6)
    plt.hist(metrics['daily_returns'], bins=50, color='blue', alpha=0.7)
    plt.axvline(-metrics['var_95'], color='red', linestyle='--', label='95% VaR')
    plt.title('Portfolio Returns Distribution & VaR')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = '2023-01-01'
    end_date = '2025-07-25'
    
    prices = download_stock_data(tickers, start_date, end_date)
    if prices is None:
        return
    
    monte_carlo_results, optimal_weights = monte_carlo_optimization(prices)
    metrics = calculate_portfolio_metrics(prices, optimal_weights)
    
    print("\nOptimal Portfolio Metrics:")
    print(f"Average Annual Return: {metrics['avg_return']*100:.2f}%")
    print(f"Annual Volatility: {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"95% Value-at-Risk: {metrics['var_95']*100:.2f}%")
    print("\nOptimal Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight*100:.2f}%")
    
    plot_portfolio_analysis(prices, metrics, optimal_weights, tickers, monte_carlo_results)

if __name__ == "__main__":
    main()
