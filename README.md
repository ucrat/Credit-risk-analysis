# Credit-risk-analysis
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to get user input for companies and portfolio weights
def get_user_input():
    # Get list of companies
    tickers_input = input("Enter the list of company tickers separated by commas (e.g., AAPL, RELIANCE.NS, TATAMOTORS.NS, HDFCBANK.NS): ")
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]
    
    # Get portfolio weights for each company
    portfolio_weights = {}
    total_weight = 0
    for ticker in tickers:
        weight = float(input(f"Enter the portfolio weight for {ticker} (e.g., 0.25 for 25%): "))
        portfolio_weights[ticker] = weight
        total_weight += weight
    
    # Check if the total weight sums to 1 (100%)
    if np.isclose(total_weight, 1.0):
        print("Portfolio weightages are valid.")
    else:
        print("Warning: Portfolio weightages do not sum to 1, but we will proceed.")
    
    return tickers, portfolio_weights

# Function to assign credit rating based on financial ratios
def assign_credit_rating(current_ratio, quick_ratio, ebitda_margin, debt_to_equity, interest_coverage, volatility):
    if current_ratio > 2.0 and quick_ratio > 1.5 and ebitda_margin > 0.2 and debt_to_equity < 0.5 and interest_coverage > 4.0 and volatility < 0.3:
        return 'AAA'
    elif current_ratio > 1.5 and quick_ratio > 1.2 and ebitda_margin > 0.15 and debt_to_equity < 1.0 and interest_coverage > 3.0 and volatility < 0.35:
        return 'AA'
    elif current_ratio > 1.2 and quick_ratio > 1.0 and ebitda_margin > 0.1 and debt_to_equity < 1.5 and interest_coverage > 2.5 and volatility < 0.4:
        return 'A'
    elif current_ratio > 1.0 and quick_ratio > 0.8 and ebitda_margin > 0.05 and debt_to_equity < 2.0 and interest_coverage > 2.0 and volatility < 0.5:
        return 'BBB'
    elif current_ratio > 0.8 and quick_ratio > 0.5 and ebitda_margin > 0.02 and debt_to_equity < 3.0 and interest_coverage > 1.5 and volatility < 0.6:
        return 'BB'
    else:
        return 'B'

# Define the mapping from Credit Rating to Probability of Default (PD)
pd_table = {
    'AAA': 0.0002,    # 0.02%
    'AA': 0.0005,     # 0.05%
    'A': 0.001,       # 0.10%
    'BBB': 0.005,     # 0.50%
    'BB': 0.02,       # 2.00%
    'B': 0.1,         # 10.00%
    'CCC': 0.25,      # 25.00%
    'D': 1.0           # 100.00%
}

# Loss Given Default (LGD) Calculation - Assume this is based on asset quality, collateral, and recovery rate
def calculate_lgd(total_debt, total_assets):
    # LGD is typically calculated as 1 - Recovery Rate, which can be estimated based on the proportion of total debt to total assets
    recovery_rate = total_assets / total_debt if total_debt != 0 else 0
    lgd = 1 - recovery_rate
    return lgd
    print(ldg)
# Function to calculate Expected Loss (EL)
def calculate_expected_loss(rating, ead, lgd, pd_table):
    # Get the PD for the given credit rating
    pd = pd_table.get(rating, 0)  # Default to 0 if rating is not found
    # Calculate Expected Loss
    el = ead * pd * lgd
    return el
    print(el)
# Mitigation Strategy Recommendations based on Expected Loss (EL)
def mitigation_strategy(el):
    if el < 1000000:
        return "Low Risk: Diversify portfolio, consider minimal hedging."
    elif el < 5000000:
        return "Moderate Risk: Hedge with Credit Derivatives, improve asset quality."
    elif el < 10000000:
        return "High Risk: Collateralization, securitization, and stress testing."
    else:
        return "Very High Risk: Consider restructuring debt, heavy hedging, and diversification."

# Get user input for companies and weights
tickers, portfolio_weights = get_user_input()

# Initialize dictionary to store company data
company_data = {}

# Fetch data for each company
for ticker in tickers:
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    income_statement = company.financials
    company_info = company.info
    
    # Safe extraction of Total Current Assets and liabilities
    current_assets = balance_sheet.loc['Current Assets'][0] if 'Current Assets' in balance_sheet.index else 0
    current_liabilities = balance_sheet.loc['Current Liabilities'][0] if 'Current Liabilities' in balance_sheet.index else 0
    inventory = balance_sheet.loc['Inventory'][0] if 'Inventory' in balance_sheet.index else 0

    # Avoid division by zero for Current and Quick Ratio
    current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
    quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 0
    print(current_ratio, quick_ratio)
    # EBIT and EBITDA
    ebit = income_statement.loc['EBIT'][0] if 'EBIT' in income_statement.index else 0
    ebitda = income_statement.loc['EBITDA'][0] if 'EBITDA' in income_statement.index else 0
    revenue = income_statement.loc['Total Revenue'][0] if 'Total Revenue' in income_statement.index else 1  # Avoid division by zero
    net_income = income_statement.loc['Net Income'][0] if 'Net Income' in income_statement.index else 0

    # EBITDA Margin
    ebitda_margin = ebitda / revenue if revenue != 0 else 0

    # Debt Ratios
    total_debt = balance_sheet.loc['Long Term Debt'][0] + balance_sheet.loc['Short Long Term Debt'][0] if 'Short Long Term Debt' in balance_sheet.index else balance_sheet.loc['Long Term Debt'][0]
    
    # Check if regularMarketPrice exists, otherwise use last closing price
    market_price = company_info.get('regularMarketPrice', company.history(period="1d")['Close'].iloc[-1])
    
    # Market Value of Equity (shares outstanding * market price)
    total_equity = company_info['sharesOutstanding'] * market_price  # Market Value of Equity

    # Debt-to-Equity Ratio
    debt_to_equity = total_debt / total_equity if total_equity != 0 else 0

    # Debt-to-EBITDA
    debt_to_ebitda = total_debt / ebitda if ebitda != 0 else 0

    # Interest Expense (from income statement) and Interest Coverage Ratio
    interest_expense = income_statement.loc['Interest Expense'][0] if 'Interest Expense' in income_statement.index else 0
    interest_coverage_ratio = ebit / interest_expense if interest_expense != 0 else 0

    # Stock Price Volatility (1-year standard deviation)
    historical_data = company.history(period="1y")['Close']
    daily_returns = historical_data.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Assign credit rating based on the ratios
    rating = assign_credit_rating(current_ratio, quick_ratio, ebitda_margin, debt_to_equity, interest_coverage_ratio, volatility)

    # Calculate LGD (Loss Given Default) based on assets and liabilities
    lgd = calculate_lgd(total_debt, total_equity + total_debt)  # Total assets are the sum of debt and equity

    # Calculate Expected Loss
    expected_loss = calculate_expected_loss(rating, total_debt, lgd, pd_table)

    # Get Mitigation Strategy
    mitigation = mitigation_strategy(expected_loss)

    # Store the data for each company
    company_data[ticker] = {
        'Rating': rating,
        'Current Ratio': current_ratio,
        'Quick Ratio': quick_ratio,
        'EBITDA Margin': ebitda_margin,
        'Debt-to-Equity': debt_to_equity,
        'Debt-to-EBITDA': debt_to_ebitda,
        'Interest Coverage Ratio': interest_coverage_ratio,
        'Volatility': volatility,
        'Expected Loss': expected_loss,
        'Mitigation Strategy': mitigation,
        'Weight': portfolio_weights[ticker],  # Add portfolio weight
        'LGD': lgd  # Add LGD value
    }

# Convert to DataFrame for visualization
df = pd.DataFrame(company_data).transpose()

# Calculate portfolio weighted expected loss
df['Weighted Expected Loss'] = df['Expected Loss'] * df['Weight']
total_expected_loss = df['Weighted Expected Loss'].sum()

# Calculate portfolio weighted volatility (simple weighted average here, more complex models can include covariance)
portfolio_volatility = np.sum(df['Volatility'] * df['Weight'])

# Output portfolio risk metrics
print(f"Total Portfolio Expected Loss: ${total_expected_loss:,.2f}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")

# Handle potential missing values by replacing them with 0
df = df.fillna(0)

# Ensure all data types are numeric for plotting
df[['Current Ratio', 'Quick Ratio', 'EBITDA Margin', 'Debt-to-Equity', 'Debt-to-EBITDA', 'Interest Coverage Ratio']] = df[['Current Ratio', 'Quick Ratio', 'EBITDA Margin', 'Debt-to-Equity', 'Debt-to-EBITDA', 'Interest Coverage Ratio']].apply(pd.to_numeric)

# Visualizations
plt.figure(figsize=(12, 6))
sns.heatmap(df[['Current Ratio', 'Quick Ratio', 'EBITDA Margin', 'Debt-to-Equity', 'Debt-to-EBITDA', 'Interest Coverage Ratio']].transpose(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Financial Ratios Heatmap')
plt.show()

# Bar plot for expected loss
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y=df['Expected Loss'], palette='Blues')
plt.title('Expected Loss for Each Company')
plt.ylabel('Expected Loss ($)')
plt.xticks(rotation=45)
plt.show()

# Box plot for volatility
plt.figure(figsize=(10, 6))
sns.boxplot(data=df['Volatility'])
plt.title('Volatility Distribution Across Companies')
plt.ylabel('Volatility')
plt.show()
print(df)

