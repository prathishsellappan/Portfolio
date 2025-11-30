
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew


def default_stocks():
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
# ------------------------ DATA FETCHING ------------------------
def fetch_data(tickers, start='2015-01-01', end='2023-01-01'):
    # auto_adjust=True ensures Close prices are adjusted
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    data = data['Close']  # Adjusted Close equivalent

    if isinstance(data, pd.Series):
        data = data.to_frame()  # ensure dataframe

    if data.empty:
        raise ValueError("No data fetched. Check tickers or date range.")

    return data


# ------------------------ RETURN CALCULATIONS ------------------------
def daily_returns(data):
    return data.pct_change().dropna()


def stats(returns):
    mean_ret = returns.mean()
    cov_mat = returns.cov()
    skewness = returns.apply(skew)
    kurt = returns.apply(kurtosis)

    return mean_ret, cov_mat, skewness, kurt


# ------------------------ PORTFOLIO METRICS ------------------------
def portfolio_perf(weights, mean_ret, cov_mat, rf_rate=0.01):
    ann_return = np.sum(weights * mean_ret) * 252
    ann_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat * 252, weights)))
    sharpe = (ann_return - rf_rate) / ann_vol
    return ann_return, ann_vol, sharpe


def neg_sharpe(weights, mean_ret, cov_mat, rf_rate=0.01):
    return -portfolio_perf(weights, mean_ret, cov_mat, rf_rate)[2]


def weight_constraint(weights):
    return np.sum(weights) - 1


# ------------------------ OPTIMIZATION ------------------------
def optimize_port(mean_ret, cov_mat, rf_rate=0.01):
    n_assets = len(mean_ret)
    initial_guess = n_assets * [1.0 / n_assets]
    constraints = {'type': 'eq', 'fun': weight_constraint}
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(
        neg_sharpe,
        initial_guess,
        args=(mean_ret, cov_mat, rf_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result


# ------------------------ RANDOM PORTFOLIOS (Frontier) ------------------------
def random_portfolios(mean_ret, cov_mat, n_port=10000, rf_rate=0.01):
    results = np.zeros((3, n_port))
    weights_rec = []

    for _ in range(n_port):
        weights = np.random.random(len(mean_ret))
        weights /= np.sum(weights)
        weights_rec.append(weights)

        r, vol, sharpe = portfolio_perf(weights, mean_ret, cov_mat, rf_rate)
        results[:, _] = [r, vol, sharpe]

    return results, weights_rec


# ------------------------ PLOTTING ------------------------
def plot_prices(data, ax):
    for col in data.columns:
        ax.plot(data.index, data[col], label=col)
    ax.set_title('Stock Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)


def plot_returns_dist(returns, ax):
    for col in returns.columns:
        ax.hist(returns[col], bins=50, alpha=0.5, label=col)
    ax.set_title('Daily Returns Distribution')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)


def plot_weights(weights, stocks, ax):
    ax.pie(weights, labels=stocks, autopct='%1.1f%%')
    ax.set_title('Optimized Portfolio Weights')
    ax.axis('equal')


def plot_frontier(mean_ret, cov_mat, opt_result, rf_rate, ax):
    results, _ = random_portfolios(mean_ret, cov_mat)

    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    ax.scatter(results[1, :], results[0, :],
               c=results[2, :], cmap='viridis', s=10, alpha=0.3)
    ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
               color='red', s=80, label='Max Sharpe')
    ax.scatter(results[1, min_vol_idx], results[0, min_vol_idx],
               color='green', s=80, label='Min Vol')

    opt_return, opt_vol, _ = portfolio_perf(opt_result.x, mean_ret, cov_mat, rf_rate)
    ax.scatter(opt_vol, opt_return, color='orange', s=100, label='Optimized')

    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True)


def plot_all(data, returns, weights, stocks, mean_ret, cov_mat, opt_result, rf_rate):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.tight_layout(pad=5)

    plot_prices(data, axes[0, 0])
    plot_returns_dist(returns, axes[0, 1])
    plot_weights(weights, stocks, axes[1, 0])
    plot_frontier(mean_ret, cov_mat, opt_result, rf_rate, axes[1, 1])

    plt.show()


# ------------------------ SUMMARY & FILE SAVE ------------------------
def summary_statistics(weights, mean_ret, cov_mat, skewness, kurt):
    r, vol, sharpe = portfolio_perf(weights, mean_ret, cov_mat)

    print("\n------ Portfolio Summary ------")
    print(f"Expected Annual Return : {r:.4f}")
    print(f"Annual Volatility      : {vol:.4f}")
    print(f"Sharpe Ratio           : {sharpe:.4f}")
    print("\nSkewness:")
    print(skewness)
    print("\nKurtosis:")
    print(kurt)


def save_to_csv(weights, filename='optimized_portfolio.csv'):
    pd.DataFrame(weights, columns=['Weights']).to_csv(filename, index=False)
    print(f"Saved optimized weights to {filename}")


# ------------------------ MAIN SCRIPT ------------------------
def main():
    print("Welcome to the Portfolio Optimization Tool!")

    stocks = default_stocks()

    data = fetch_data(stocks)
    returns = daily_returns(data)
    mean_ret, cov_mat, skewness, kurt = stats(returns)

    opt_result = optimize_port(mean_ret, cov_mat)

    summary_statistics(opt_result.x, mean_ret, cov_mat, skewness, kurt)

    plot_all(data, returns, opt_result.x, stocks, mean_ret, cov_mat, opt_result, 0.01)

    save_to_csv(opt_result.x)


if __name__ == "__main__":
    main()