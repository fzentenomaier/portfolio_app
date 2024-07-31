import pandas as pd
import numpy as np
import streamlit as st
import datetime
import yfinance as yf
# Plotly for Data Visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
        
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def get_sp500():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(sp500_url)
    sp500_components = sp500_table[0]
    sp500_dict = dict(zip(sp500_components['Security'].tolist(),
                          sp500_components['Symbol'].tolist()))
    sp500_dict2 = dict(zip(sp500_components['Symbol'].tolist(),
                          sp500_components['Security'].tolist()))
    list_names = sp500_components['Security'].tolist() + sp500_components['Symbol'].tolist()
    sp500_components.rename(columns={"GICS Sector": "Industries"}, inplace=True)
    return sp500_components, sp500_dict, sp500_dict2, list_names


def portfolio_returns(tickers_and_values, start_date, end_date, benchmark):
    # Obtaining tickers data with yfinance
    df = yf.download(tickers=list(tickers_and_values.keys()),
                     start=start_date, end=end_date)

    # Checking if there is data available in the given date range
    if isinstance(df.columns, pd.MultiIndex):
        missing_data_tickers = []
        for ticker in tickers_and_values.keys():
            first_valid_index = df['Adj Close'][ticker].first_valid_index()
            if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
                missing_data_tickers.append(ticker)

        RED_BOLD_TEXT = '\033[1;31m'
        RESET_TEXT = '\033[0m'

        if missing_data_tickers:
            print(f"{RED_BOLD_TEXT}\n No data available for the following tickers starting from {start_date}: {', '.join(missing_data_tickers)}{RESET_TEXT}")
            return
    else:
        # For a single ticker, simply check the first valid index
        first_valid_index = df['Adj Close'].first_valid_index()
        if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
            print(f"{RED_BOLD_TEXT}\n No data available for the ticker starting from {start_date}{RESET_TEXT}")
            return
    
    # Calculating portfolio value
    total_portfolio_value = sum(tickers_and_values.values())

    # Calculating the weights for each security in the portfolio
    tickers_weights = {ticker: value / total_portfolio_value for ticker, value in tickers_and_values.items()}

    # Checking if dataframe has MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close'].fillna(df['Close']) # If 'Adjusted Close' is not available, use 'Close'

    # Checking if there are more than just one security in the portfolio
    if len(tickers_weights) > 1:
        weights = list(tickers_weights.values()) # Obtaining weights
        weighted_returns = df.pct_change().mul(weights, axis = 1) # Computed weighted returns
        port_returns = weighted_returns.sum(axis=1) # Sum weighted returns to build portfolio returns
    # If there is only one security in the portfolio...
    else:
        df = df['Adj Close'].fillna(df['Close'])  # Obtaining 'Adjusted Close'. If not available, use 'Close'
        port_returns = df.pct_change() # Computing returns without weights

    # Obtaining benchmark data with yfinance
    benchmark_df = yf.download(benchmark, 
                               start=start_date, end=end_date) 
    # Obtaining 'Adjusted Close'. If not available, use 'Close'.
    benchmark_df = benchmark_df['Adj Close'].fillna(benchmark_df['Close'])

    # Computing benchmark returns
    benchmark_returns = benchmark_df.pct_change()


    # Plotting a pie plot
    fig = go.Figure(data=[go.Pie(
        labels=list(tickers_weights.keys()), # Obtaining tickers 
        values=list(tickers_weights.values()), # Obtaining weights
        hoverinfo='label+percent', 
        textinfo='label+percent',
        hole=.65,
        marker=dict(colors=px.colors.qualitative.G10)
    )])

    # Defining layout
    fig.update_layout(title={'text': '<b>Portfolio Allocation</b>'}, height=550)

    # Running function to compare portfolio and benchmark
    fig2 = portfolio_vs_benchmark(port_returns, benchmark_returns)    

    # If we have more than one security in the portfolio, 
    # we run function to evaluate each security individually
    if len(tickers_weights) > 1:
        fig1 = perform_portfolio_analysis(df, tickers_weights)
    # Displaying Portfolio vs Benchmark plot    
    return port_returns, benchmark_returns, fig, fig1, fig2
    
    
def perform_portfolio_analysis(df, tickers_weights):
    # Starting DataFrame and Series 
    individual_cumsum = pd.DataFrame()
    individual_vol = pd.Series(dtype=float)
    individual_sharpe = pd.Series(dtype=float)


    # Iterating through tickers and weights in the tickers_weights dictionary
    for ticker, weight in tickers_weights.items():
        if ticker in df.columns: # Confirming that the tickers are available
            individual_returns = df[ticker].pct_change() # Computing individual daily returns for each ticker
            individual_cumsum[ticker] = ((1 + individual_returns).cumprod() - 1) * 100 # Computing cumulative returns over the period for each ticker 
            vol = (individual_returns.std() * np.sqrt(252)) * 100 # Computing annualized volatility
            individual_vol[ticker] = vol # Adding annualized volatility for each ticker
            individual_excess_returns = individual_returns - 0.01 / 252 # Computing the excess returns
            sharpe = (individual_excess_returns.mean() / individual_returns.std() * np.sqrt(252)).round(2) # Computing Sharpe Ratio
            individual_sharpe[ticker] = sharpe # Adding Sharpe Ratio for each ticker

            # Creating subplots for comparison across securities
            fig1 = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.2,
                            column_titles=['Historical Performance Assets', 'Risk-Reward'],
                            column_widths=[.50, .50],
                            shared_xaxes=False, shared_yaxes=False)
        
    # Adding the historical returns for each ticker on the first subplot    
    for ticker in individual_cumsum.columns:
        fig1.add_trace(go.Scatter(x=individual_cumsum.index,
                                  y=individual_cumsum[ticker],
                                  mode = 'lines',
                                  name = ticker,
                                  hovertemplate = '%{y:.2f}%',
                                  showlegend=False),
                            row=1, col=1)

    # Defining colors for markers on the second subplot
    sharpe_colors = [individual_sharpe[ticker] for ticker in individual_cumsum.columns]

    # Adding markers for each ticker on the second subplot
    fig1.add_trace(go.Scatter(x=individual_vol.tolist(),
                              y=individual_cumsum.iloc[-1].tolist(),
                              mode='markers+text',
                              marker=dict(size=75, color = sharpe_colors, 
                                          colorscale = 'Bluered_r',
                                          colorbar=dict(title='Sharpe Ratio'),
                                          showscale=True),
                              name = 'Returns',
                              text = individual_cumsum.columns.tolist(),
                              textfont=dict(color='white'),
                              showlegend=False,
                              hovertemplate = '%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                              textposition='middle center'),
                        row=1, col=2)
            
    # Updating layout
    fig1.update_layout(#title={'text': f'<b>Portfolio Analysis</b>'},
                       template = 'plotly_white',
                       height = 700, width = 1000,
                       hovermode = 'x unified')
        
    fig1.update_yaxes(title_text='Returns (%)', col=1)
    fig1.update_yaxes(title_text='Returns (%)', col = 2)
    fig1.update_xaxes(title_text = 'Date', col = 1)
    fig1.update_xaxes(title_text = 'Annualized Volatility (%)', col =2)
            
    return fig1


def portfolio_vs_benchmark(port_returns, benchmark_returns):
    # Computing the cumulative returns for the portfolio and the benchmark
    portfolio_cumsum = (((1 + port_returns).cumprod() - 1) * 100).round(2)
    benchmark_cumsum = (((1 + benchmark_returns).cumprod() - 1) * 100).round(2)

    # Computing the annualized volatility for the portfolio and the benchmark
    port_vol = ((port_returns.std() * np.sqrt(252)) * 100).round(2)
    benchmark_vol = ((benchmark_returns.std() * np.sqrt(252)) * 100).round(2)

    # Computing Sharpe Ratio for the portfolio and the benchmark
    excess_port_returns = port_returns - 0.01 / 252
    port_sharpe = (excess_port_returns.mean() / port_returns.std() * np.sqrt(252)).round(2)
    exces_benchmark_returns = benchmark_returns - 0.01 / 252
    benchmark_sharpe = (exces_benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)).round(2)

    # Creating a subplot to compare portfolio performance with the benchmark
    fig2 = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.2,
                        column_titles=['Cumulative Returns', 'Portfolio Risk-Reward'],
                        column_widths=[.50, .50],
                        shared_xaxes=False, shared_yaxes=False)

    # Adding the cumulative returns for the portfolio
    fig2.add_trace(go.Scatter(x=portfolio_cumsum.index, 
                             y = portfolio_cumsum,
                             mode = 'lines', name = 'Portfolio', showlegend=False,
                             hovertemplate = '%{y:.2f}%'),
                             row=1,col=1)
    
    # Adding the cumulative returns for the benchmark
    fig2.add_trace(go.Scatter(x=benchmark_cumsum.index, 
                             y = benchmark_cumsum,
                             mode = 'lines', name = 'Benchmark', showlegend=False,
                             hovertemplate = '%{y:.2f}%'),
                             row=1,col=1)
    

    # Creating risk-reward plot for the benchmark and the portfolio
    fig2.add_trace(go.Scatter(x = [port_vol, benchmark_vol], y = [portfolio_cumsum.iloc[-1], benchmark_cumsum.iloc[-1]],
                             mode = 'markers+text', 
                             marker=dict(size = 75, 
                                         color = [port_sharpe, benchmark_sharpe],
                                         colorscale='Bluered_r',
                                         colorbar=dict(title='Sharpe Ratio'),
                                         showscale=True),
                             name = 'Returns', 
                             text=['Portfolio', 'Benchmark'], textposition='middle center',
                             textfont=dict(color='white'),
                             hovertemplate = '%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                             showlegend=False),
                             row = 1, col = 2)
    
    
    # Configuring layout
    fig2.update_layout(#title={'text': f'<b>Portfolio vs Benchmark</b>'},
                      template = 'plotly_white',
                      height = 700, width = 1000,
                      hovermode = 'x unified')
    
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col=1)
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col = 2)
    fig2.update_xaxes(title_text = 'Date', col = 1)
    fig2.update_xaxes(title_text = 'Annualized Volatility (%)', col =2)

    return fig2 # Returning subplots


def get_portfolio_opt(stocks, start, end):
    
    # Step 1: Download stock data from Yahoo Finance
    data = yf.download(stocks, start, end)['Close']
    
    # Step 2: Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Step 3: Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.01  # Assuming a 1% risk-free rate
    
    # Step 4: Portfolio performance function
    def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (returns - risk_free_rate) / std
        return returns, std, sharpe_ratio
    
    # Function to minimize
    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[1]
    
    # Function to minimize negative Sharpe ratio
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]
    
    # Step 5: Set constraints and bounds
    num_assets = len(stocks)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess
    init_guess = num_assets * [1. / num_assets]
    
    # Step 6: Perform the optimization for minimum volatility
    optimized_vol_results = minimize(minimize_volatility, init_guess, args=(mean_returns, cov_matrix),
                                     method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_vol_weights = optimized_vol_results.x
    
    # Perform the optimization for maximum Sharpe ratio
    optimized_sharpe_results = minimize(negative_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                                        method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_sharpe_weights = optimized_sharpe_results.x
    
    # Step 7: Generate portfolios
    def generate_portfolios(mean_returns, cov_matrix, num_portfolios=5000):
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            portfolio_return, portfolio_std, portfolio_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
            
            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = portfolio_sharpe
            
            weights_record.append(weights)
            
        return results, weights_record
    
    # Step 8: Generate Efficient Frontier
    num_portfolios = 5000
    results, weights = generate_portfolios(mean_returns, cov_matrix, num_portfolios)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot Efficient Frontier
    fig.add_trace(go.Scatter(
        x=results[0, :],
        y=results[1, :],
        mode='markers',
        marker=dict(color=results[2, :], colorscale='YlGnBu', colorbar=dict(title='Sharpe Ratio')),
        name='Portfolios'
    ))
    
    # Highlight the optimized portfolio for minimum volatility
    optimized_vol_return, optimized_vol_volatility, _ = portfolio_performance(optimized_vol_weights, mean_returns, cov_matrix, risk_free_rate)
    fig.add_trace(go.Scatter(
        x=[optimized_vol_volatility],
        y=[optimized_vol_return],
        mode='markers',
        marker=dict(color='green', size=10, symbol='star'),
        name='Min Volatility Portfolio'
    ))
    
    # Highlight the optimized portfolio for maximum Sharpe ratio
    optimized_sharpe_return, optimized_sharpe_volatility, _ = portfolio_performance(optimized_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    fig.add_trace(go.Scatter(
        x=[optimized_sharpe_volatility],
        y=[optimized_sharpe_return],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Max Sharpe Ratio Portfolio'
    ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Standard Deviation)',
        yaxis_title='Expected Returns',
        legend=dict(
            x=-0.2,  # Adjust this value to move the legend further left
            y=1,  # Adjust this value to move the legend vertically
            traceorder='normal',
            orientation='v'
        )
    )
    def filter_small_values(values, threshold=0.01):
        return [value if value >= threshold else 0 for value in values]
    optimized_vol_weights = filter_small_values(optimized_vol_weights, threshold=0.01)
    optimized_sharpe_weights = filter_small_values(optimized_sharpe_weights, threshold=0.01)

    # Return Plotly figure and portfolio details
    port_min = [optimized_vol_weights, optimized_vol_return, optimized_vol_volatility]
    port_max = [optimized_sharpe_weights, optimized_sharpe_return, optimized_sharpe_volatility]
    
    return fig, port_min, port_max


def plot_portfolio_allocation(port_min, port_max, labels):
    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    
    # Add pie charts to the subplots
    fig.add_trace(go.Pie(
        labels=labels, 
        values=port_min[0], 
        name="Min Volatility",
        hole=.65,  # Use hole to create a donut chart
        hoverinfo='label+percent', 
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.G10)
    ), row=1, col=1)
    
    
    fig.add_trace(go.Pie(
        labels=labels, 
        values=port_max[0], 
        name="Max Sharpe Ratio",
        hole=.65,  # Use hole to create a donut chart
        hoverinfo='label+percent', 
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.G10)
    ), row=1, col=2)
    
    # Update layout for subplots
    fig.update_layout(
        title_text="Optimized Portfolios",
        height=550,
        showlegend=True,  # Show legend if needed
        annotations=[
            dict(
                text='Min Vol',
                x=0.18,
                y=0.5,
                font_size=20,
                showarrow=False
            ),
            dict(
                text='Max Sharpe',
                x=0.82,
                y=0.5,
                font_size=20,
                showarrow=False
            )
        ]
    )
    
    return fig


def portfolio_returns_extra(tickers_and_values, start_date, end_date, benchmark, min_vol_weights, max_sharpe_weights):
    # Obtaining tickers data with yfinance
    tickers = list(set(list(tickers_and_values.keys()) + list(min_vol_weights.keys()) + list(max_sharpe_weights.keys())))
    df = yf.download(tickers=tickers, start=start_date, end=end_date)

    # Checking if there is data available in the given date range
    if isinstance(df.columns, pd.MultiIndex):
        missing_data_tickers = []
        for ticker in tickers:
            first_valid_index = df['Adj Close'][ticker].first_valid_index()
            if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
                missing_data_tickers.append(ticker)

        RED_BOLD_TEXT = '\033[1;31m'
        RESET_TEXT = '\033[0m'

        if missing_data_tickers:
            print(f"{RED_BOLD_TEXT}\n No data available for the following tickers starting from {start_date}: {', '.join(missing_data_tickers)}{RESET_TEXT}")
            return
    else:
        # For a single ticker, simply check the first valid index
        first_valid_index = df['Adj Close'].first_valid_index()
        if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
            print(f"{RED_BOLD_TEXT}\n No data available for the ticker starting from {start_date}{RESET_TEXT}")
            return
    
    # Calculating portfolio returns based on weights
    def calculate_returns_extra(df, weights):
        # Checking if dataframe has MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Adj Close'].fillna(df['Close'])  # If 'Adjusted Close' is not available, use 'Close'

        # Calculating weighted returns
        total_value = sum(weights.values())
        weight_proportions = {ticker: value / total_value for ticker, value in weights.items()}
        if len(weight_proportions) > 1:
            weighted_returns = df.pct_change().mul(list(weight_proportions.values()), axis=1)
            portfolio_returns = weighted_returns.sum(axis=1)
        else:
            ticker = next(iter(weight_proportions))
            portfolio_returns = df[ticker].pct_change()

        return portfolio_returns

    # Calculating returns for each portfolio
    port_returns = calculate_returns_extra(df, tickers_and_values)
    min_vol_returns = calculate_returns_extra(df, min_vol_weights)
    max_sharpe_returns = calculate_returns_extra(df, max_sharpe_weights)

    # Obtaining benchmark data with yfinance
    benchmark_df = yf.download(benchmark, start=start_date, end=end_date) 
    benchmark_df = benchmark_df['Adj Close'].fillna(benchmark_df['Close'])  # If 'Adjusted Close' is not available, use 'Close'

    # Computing benchmark returns
    benchmark_returns = benchmark_df.pct_change()

    # Plotting a pie plot
    fig = go.Figure(data=[go.Pie(
        labels=list(tickers_and_values.keys()),  # Obtaining tickers 
        values=list(tickers_and_values.values()),  # Obtaining weights
        hoverinfo='label+percent', 
        textinfo='label+percent',
        hole=.65,
        marker=dict(colors=px.colors.qualitative.G10)
    )])

    # Defining layout
    fig.update_layout(title={'text': '<b>Portfolio Allocation</b>'}, height=550)

    # Running function to compare portfolio and benchmark
    fig2 = portfolio_vs_benchmark_extra(port_returns, benchmark_returns, min_vol_returns, max_sharpe_returns)    

    fig1 = 0
    # Displaying Portfolio vs Benchmark plot    
    return port_returns, benchmark_returns, fig, fig1, fig2
    
    
def perform_portfolio_analysis(df, tickers_weights):
    # Starting DataFrame and Series 
    individual_cumsum = pd.DataFrame()
    individual_vol = pd.Series(dtype=float)
    individual_sharpe = pd.Series(dtype=float)

    # Iterating through tickers and weights in the tickers_weights dictionary
    for ticker, weight in tickers_weights.items():
        if ticker in df.columns:  # Confirming that the tickers are available
            individual_returns = df[ticker].pct_change()  # Computing individual daily returns for each ticker
            individual_cumsum[ticker] = ((1 + individual_returns).cumprod() - 1) * 100  # Computing cumulative returns over the period for each ticker 
            vol = (individual_returns.std() * np.sqrt(252)) * 100  # Computing annualized volatility
            individual_vol[ticker] = vol  # Adding annualized volatility for each ticker
            individual_excess_returns = individual_returns - 0.01 / 252  # Computing the excess returns
            sharpe = (individual_excess_returns.mean() / individual_returns.std() * np.sqrt(252)).round(2)  # Computing Sharpe Ratio
            individual_sharpe[ticker] = sharpe  # Adding Sharpe Ratio for each ticker

    # Creating subplots for comparison across securities
    fig1 = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                         column_titles=['Historical Performance Assets', 'Risk-Reward'],
                         column_widths=[.50, .50],
                         shared_xaxes=False, shared_yaxes=False)
        
    # Adding the historical returns for each ticker on the first subplot    
    for ticker in individual_cumsum.columns:
        fig1.add_trace(go.Scatter(x=individual_cumsum.index,
                                  y=individual_cumsum[ticker],
                                  mode='lines',
                                  name=ticker,
                                  hovertemplate='%{y:.2f}%',
                                  showlegend=False),
                       row=1, col=1)

    # Defining colors for markers on the second subplot
    sharpe_colors = [individual_sharpe[ticker] for ticker in individual_cumsum.columns]

    # Adding markers for each ticker on the second subplot
    fig1.add_trace(go.Scatter(x=individual_vol.tolist(),
                              y=individual_cumsum.iloc[-1].tolist(),
                              mode='markers+text',
                              marker=dict(size=75, color=sharpe_colors, 
                                          colorscale='Bluered_r',
                                          colorbar=dict(title='Sharpe Ratio'),
                                          showscale=True),
                              name='Returns',
                              text=individual_cumsum.columns.tolist(),
                              textfont=dict(color='white'),
                              showlegend=False,
                              hovertemplate='%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                              textposition='middle center'),
                       row=1, col=2)
            
    # Updating layout
    fig1.update_layout(template='plotly_white',
                       height=700, width=1000,
                       hovermode='x unified')
        
    fig1.update_yaxes(title_text='Returns (%)', col=1)
    fig1.update_yaxes(title_text='Returns (%)', col=2)
    fig1.update_xaxes(title_text='Date', col=1)
    fig1.update_xaxes(title_text='Annualized Volatility (%)', col=2)
            
    return fig1


def portfolio_vs_benchmark_extra(port_returns, benchmark_returns, min_vol_returns, max_sharpe_returns):
    # Function to compute cumulative returns, volatility, and Sharpe ratio
    def compute_metrics(returns):
        cumsum = (((1 + returns).cumprod() - 1) * 100).round(2)
        vol = ((returns.std() * np.sqrt(252)) * 100).round(2)
        excess_returns = returns - 0.01 / 252
        sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252)).round(2)
        return cumsum, vol, sharpe

    # Compute metrics for each portfolio
    port_cumsum, port_vol, port_sharpe = compute_metrics(port_returns)
    benchmark_cumsum, benchmark_vol, benchmark_sharpe = compute_metrics(benchmark_returns)
    min_vol_cumsum, min_vol_vol, min_vol_sharpe = compute_metrics(min_vol_returns)
    max_sharpe_cumsum, max_sharpe_vol, max_sharpe_sharpe = compute_metrics(max_sharpe_returns)

    # Create a subplot to compare portfolio performance with the benchmark
    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                         column_titles=['Cumulative Returns', 'Portfolio Risk-Reward'],
                         column_widths=[.50, .50],
                         shared_xaxes=False, shared_yaxes=False)

    # Add cumulative returns for each portfolio
    fig2.add_trace(go.Scatter(x=port_cumsum.index, y=port_cumsum, mode='lines', name='Original Portfolio', hovertemplate='%{y:.2f}%'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=benchmark_cumsum.index, y=benchmark_cumsum, mode='lines', name='Benchmark', hovertemplate='%{y:.2f}%'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=min_vol_cumsum.index, y=min_vol_cumsum, mode='lines', name='Min Volatility Portfolio', hovertemplate='%{y:.2f}%'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=max_sharpe_cumsum.index, y=max_sharpe_cumsum, mode='lines', name='Max Sharpe Portfolio', hovertemplate='%{y:.2f}%'), row=1, col=1)

    # Create risk-reward plot for each portfolio
    fig2.add_trace(go.Scatter(x=[port_vol, benchmark_vol, min_vol_vol, max_sharpe_vol], 
                              y=[port_cumsum.iloc[-1], benchmark_cumsum.iloc[-1], min_vol_cumsum.iloc[-1], max_sharpe_cumsum.iloc[-1]],
                              mode='markers+text', 
                              marker=dict(size=75, 
                                          color=[port_sharpe, benchmark_sharpe, min_vol_sharpe, max_sharpe_sharpe],
                                          colorscale='Bluered_r',
                                          colorbar=dict(title='Sharpe Ratio'),
                                          showscale=True),
                              text=['Original Portfolio', 'Benchmark', 'Min Volatility Portfolio', 'Max Sharpe Portfolio'], 
                              textposition='middle center',
                              textfont=dict(color='white'),
                              hovertemplate='%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                              showlegend=False),
                       row=1, col=2)
    
    # Configuring layout
    fig2.update_layout(template='plotly_white',
                       height=700, width=1000,
                       hovermode='x unified')
    
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col=1)
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col=2)
    fig2.update_xaxes(title_text='Date', col=1)
    fig2.update_xaxes(title_text='Annualized Volatility (%)', col=2)

    return fig2  # Returning subplots


#%% Streamlit
def main():
    st.set_page_config(layout="wide")
    sp500_com, dic_sp500, dic_sp500_2, list_500 = get_sp500()
    st.title("Stock Portfolio App (S&P 500)")
    st.markdown("""Welcome to the Stock Portfolio App, your comprehensive tool for managing and optimizing stock investments. This app allows you to input your stock portfolio, specify the initial investment amount, and set the investment period. You can also assign weights to each stock to tailor your portfolio to your preferences.\n\n
                \n\nUsing `yfinance` for accurate financial data retrieval and `Plotly` for interactive visualizations, our app provides detailed insights into your portfolio's performance and compares it with the S&P 500 benchmark.\n\n
                \n\nWe're also working on implementing the **Markowitz Mean-Variance model** to help you optimize your portfolio's risk and return. \n\n\n\n""")

    col1,col3,col4,col5 = st.columns([4,2,2,1])
    
    with col1:
        stocks_selected = st.multiselect('Select the **Stocks** in your portfolio', list_500, ['AAPL', 'HPQ', 'NVDA'])
    with col3:
        initial_amount_selected = st.number_input("Initial Amount (USD)", value=1000, placeholder="e.g 1000")
    with col4:
        time_period = ['<select>', 2,3,4,5,6,7,8,9]
        default_ix = time_period.index(7)
        time_period_selected = st.selectbox('**Period** (Years)', time_period, index=default_ix)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=time_period_selected*365)
    #format
    end = end.strftime('%Y-%m-%d')
    start = start.strftime('%Y-%m-%d')
    list_stock_ticker = []
    list_stock_security = []
    if len(stocks_selected) > 1:
        
    
        for stock_selected in stocks_selected:
            if stock_selected in (sp500_com['Security'].tolist()):
                stock_security = stock_selected
                stock_ticker = dic_sp500[stock_selected]
            if stock_selected in (sp500_com['Symbol'].tolist()):
                stock_security = dic_sp500_2[stock_selected] 
                stock_ticker = stock_selected
            list_stock_ticker.append(stock_ticker)
            list_stock_security.append(stock_security)
        st.markdown(f"<h1 style='text-align: center;'> Select the Weights of your Portfolio </h1>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([5,1,5])
        with col1:
            values_stocks = []
            number_stocks = len(list_stock_ticker)
            for i in range(len(list_stock_ticker)):
                if not values_stocks:
                    final_value = 50
                values = st.slider(f" {list_stock_security[i]} ({list_stock_ticker[i]}) ", 0, 100, (final_value))
                number_stocks = number_stocks - 1
                values_stocks.append(values)
                if not number_stocks == 0:
                    final_value = int((100 - sum(values_stocks))/number_stocks)
                    
            if sum(values_stocks) != 100 or any(x < 0 for x in values_stocks):
                st.error('The total weight of the selected stocks exceeds 100%', icon="🚨")
    
        if sum(values_stocks) == 100:
            #st.write('listo')
            values_stocks_new = [(x/100)*initial_amount_selected for x in values_stocks]
            tickers = dict(zip(list_stock_ticker, values_stocks_new))
            port_returns, benchmark_returns, fig, fig1, fig2 = portfolio_returns(tickers, start, end, '^GSPC')
        else:
            st.error('The total weight of the selected stocks exceeds 100%', icon="🚨")
            
        st.markdown(f"<h1 style='text-align: center;'> Portfolio Analysis </h1>", unsafe_allow_html=True)
        with col3:
            fig.update_layout(showlegend=False)
            st.write(fig)
        
        col1,col2,col3 = st.columns([1,15,1])
        with col2:
            st.write(fig1)
            st.markdown(f"<h1 style='text-align: center;'> Portfolio vs Benchmark (S&P500) </h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 14px;'> For this case, the benchmark is the ^GSPC, which represents the S&P 500 index </p>", unsafe_allow_html=True)
            st.write(fig2)
        st.divider()
        st.markdown(f"<h1 style='text-align: center;'>  MVO and Markowitz’s Efficient Frontier</h1>", unsafe_allow_html=True)
    
        with st.expander('**Explanation**'):
            # Introduction text
            st.markdown("""
                The Mean-Variance Optimization (MVO) and Markowitz’s Efficient Frontier, foundational concepts in modern portfolio theory. Developed by Harry Markowitz in 1952, this method assists investors in constructing portfolios that optimize expected returns for a given level of risk. The Efficient Frontier represents the set of optimal portfolios offering the highest expected return for a defined level of risk or the lowest risk for a given level of expected return.
            """)
            
            # Expected Portfolio Return formula
            st.markdown("""
                ### Expected Portfolio Return
                The expected return of a portfolio is calculated as the weighted sum of the expected returns of the individual assets in the portfolio:
            """)
            expected_return_formula = r'\text{Expected Return} = \sum_{i=1}^{n} w_i \mu_i'
            st.latex(expected_return_formula)
            
            # Portfolio Volatility formula
            st.markdown("""
                ### Portfolio Volatility
                The risk (volatility) of a portfolio is determined using the covariance matrix of the asset returns:
            """)
            volatility_formula = r'\text{Volatility} = \sqrt{ \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} }'
            st.latex(volatility_formula)
            
            # Sharpe Ratio formula
            st.markdown("""
                ### Sharpe Ratio
                The Sharpe ratio is a measure of risk-adjusted return:
            """)
            sharpe_ratio_formula = r'\text{Sharpe Ratio} = \frac{ \text{Expected Return} - \text{Risk-Free Rate} }{ \text{Volatility} }'
            st.latex(sharpe_ratio_formula)
        if st.button("Run the Optimization Model"):
         
            fig, port_min, port_max = get_portfolio_opt(list_stock_ticker, start, end)
            
            col1, col2, col3 = st.columns([5,1,5])
            with col1:
                st.write(fig)
            
            with col3:
                fig = plot_portfolio_allocation(port_min, port_max, list_stock_ticker)
                fig.update_layout(showlegend=False)
                st.write(fig)
            #Min Vol
            values_stocks_min = [int(x*initial_amount_selected) for x in port_min[0]]
            tickers_min = dict(zip(list_stock_ticker, values_stocks_min))
            #Max Sharpe
            values_stocks_max = [int(x*initial_amount_selected) for x in port_max[0]]
            tickers_max = dict(zip(list_stock_ticker, values_stocks_max))
            
            port_returns, benchmark_returns, fig, fig1, fig2 = portfolio_returns_extra(tickers, start, end, '^GSPC', tickers_min, tickers_max)
            fig2.update_layout(showlegend=False)
            st.markdown(f"<h1 style='text-align: center;'> Evaluation of All Portfolios </h1>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,5,1])
            with col2:
                st.write(fig2)
                    
        else:
                st.subheader("Select a minimum of two stocks.")

    st.markdown("##")
    st.markdown("##")
    st.write("© Copyright 2024 Felipe Zenteno  All rights reserved.")
if __name__ == '__main__':
    main()
