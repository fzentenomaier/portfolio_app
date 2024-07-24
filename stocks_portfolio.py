import pandas as pd
import numpy as np
import streamlit as st
import datetime
import yfinance as yf
# Plotly for Data Visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

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
        default_ix = time_period.index(5)
        time_period_selected = st.selectbox('**Period** (Years)', time_period, index=default_ix)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=time_period_selected*365)
    #format
    end = end.strftime('%Y-%m-%d')
    start = start.strftime('%Y-%m-%d')
    list_stock_ticker = []
    list_stock_security = []

    for stock_selected in stocks_selected:
        if stock_selected in (sp500_com['Security'].tolist()):
            stock_security = stock_selected
            stock_ticker = dic_sp500[stock_selected]
        if stock_selected in (sp500_com['Symbol'].tolist()):
            stock_security = dic_sp500_2[stock_selected] 
            stock_ticker = stock_selected
        list_stock_ticker.append(stock_ticker)
        list_stock_security.append(stock_security)
    st.markdown(f"<h1 style='text-align: center;'> Select the Weights for your Portfolio </h1>", unsafe_allow_html=True)
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
        st.write(fig2)

if __name__ == '__main__':
    main()
