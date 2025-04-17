# TheCodeyBunch_Buffer_6.0 Loop Buffer 6.0 Project
# Importing necessary libraries
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize
# Sample Data - To be Replaced with actual CSV or database queries
stock_data = pd.DataFrame({
    'Company': ['TCS', 'Infosys', 'Wipro'],
    'CurrentPrice': [3600, 1500, 580],
    'High': [3650, 1520, 600],
    'Low': [3500, 1450, 560],
    'Return': [0.07, 0.05, 0.04],
    'Volatility': [0.15, 0.12, 0.11]
})

user_holdings = pd.DataFrame({
    'UserID': [1, 1, 1],
    'Company': ['TCS', 'Infosys', 'Wipro'],
    'Quantity': [10, 20, 30],
    'BuyPrice': [3400, 1400, 550]
})

# ----- Data Structures USED -----
# Hash Table: Stock lookup, ensures much faster access to for lookup
stock_hash = {row['Company']: row for _, row in stock_data.iterrows()}

# BST: Sorting by return (simple simulation using list sort), easily sort out shares
sorted_by_return = stock_data.sort_values(by='Return', ascending=False)

# Queue: Rolling window for 2-day moving average simulation for the prices fluctuations
price_history = {
    'TCS': deque([3500, 3600], maxlen=2),
    'Infosys': deque([1450, 1500], maxlen=2),
    'Wipro': deque([560, 580], maxlen=2)
}

moving_avg = {company: sum(prices)/len(prices) for company, prices in price_history.items()}

# ----- Mean-Variance Optimization ----- PARAMETER/CRITERIA FOR INDIVIDUAL PORTFOLIO OPTIMISATION
def mean_variance_optimization(returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(returns)

    def portfolio_performance(weights):
        ret = np.dot(weights, returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return -sharpe  # negative for minimization

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_performance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

returns = stock_data['Return'].values
cov_matrix = np.diag(stock_data['Volatility'].values ** 2)
optimal_weights = mean_variance_optimization(returns, cov_matrix)

# ----- ML Clustering for (Risk Profiles) -----
risk_profiles = stock_data[['Return', 'Volatility']]
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(risk_profiles)
stock_data['RiskCluster'] = clusters

# ----- Portfolio Analysis for User -----
def analyze_user_portfolio(user_id):
    user_data = user_holdings[user_holdings['UserID'] == user_id]
    print(f"\nPortfolio for User {user_id}:")
    total_value = 0
    for _, row in user_data.iterrows():
        company = row['Company']
        quantity = row['Quantity']
        current_price = stock_hash[company]['CurrentPrice']
        value = quantity * current_price
        total_value += value
        print(f"{company}: {quantity} shares @ ₹{current_price} = ₹{value}")
    print(f"Total Portfolio Value: ₹{total_value}")

# ----- Recommendation based on MVO -----
def show_optimized_portfolio():
    print("\nOptimized Portfolio Allocation:")
    for i, row in stock_data.iterrows():
        print(f"{row['Company']}: {round(optimal_weights[i]*100, 2)}%")

# Run analysis and show results
analyze_user_portfolio(1)
show_optimized_portfolio()

# Show Cluster Plot for Risks
plt.scatter(stock_data['Return'], stock_data['Volatility'], c=stock_data['RiskCluster'])
plt.xlabel('Return')
plt.ylabel('Volatility')
plt.title('Risk Clusters of Stocks')
plt.grid(True)
plt.show()
labels = stock_data['Company']
sizes = optimal_weights * 100
colors = ['#ff9999', '#66b3ff', '#99ff99']
# Final Piechart Summarizing all Values
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
plt.title('Optimized Portfolio Allocation (MVO)')
plt.tight_layout()
plt.show()
