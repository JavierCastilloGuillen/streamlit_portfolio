import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd


current_year = dt.datetime.now().year

def create_chart(stocks, weights):
    fig = go.Figure(
        data=[go.Pie(labels=stocks, values=weights, hole=0.3)]
    )
    
    fig.update_traces(
        hoverinfo='label+percent+name',
        textinfo='percent',
        textfont_size=18,
        marker=dict(
            colors=px.colors.sequential.RdBu,  # Using a sequential color palette for better aesthetics
            line=dict(color='#FFFFFF', width=2)  # White border for a cleaner look
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Stock Portfolio Allocation',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'# Customizing title font
        },
        annotations=[dict(
            text='Stocks', 
            x=0.5, 
            y=0.5, 
            font_size=20, 
            showarrow=False
        )],
        showlegend=True,  # Enabling legend for better readability
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )    
    return fig


def create_returns_plot(returns):
    fig = px.line(returns, title='Simple Returns Plot')
    return fig

def get_data(symbols, startyear):
    startmonth = 1
    startday = 1
    start = dt.datetime(startyear, startmonth, startday)
    now = dt.datetime.now()
    
    valid_symbols = []
    invalid_symbols = []
    data_frames = []

    for symbol in symbols:
        try:
            data = yf.download(symbol, start, now)
            if not data.empty:
                valid_symbols.append(symbol)
                data_frames.append(data['Adj Close'].rename(symbol))
            else:
                invalid_symbols.append(symbol)
        except Exception as e:
            invalid_symbols.append(symbol)

    if invalid_symbols:
        st.sidebar.warning(f"The following symbols are invalid and will be ignored: {', '.join(invalid_symbols)}")

    if data_frames:
        data = pd.concat(data_frames, axis=1)
        return data
    else:
        st.sidebar.error("No valid symbols provided. Please enter valid stock symbols.")
        return pd.DataFrame()


def create_ef_ft(pvols, prets):
    sharpe_ratios = prets / pvols
    fig = px.scatter(
        x=pvols, 
        y=prets, 
        color=sharpe_ratios,
        labels={'x': 'Expected Volatility', 'y': 'Expected Return', 'color': 'Sharpe Ratio'},
        title='Monte Carlo Simulation Portfolios',
        color_continuous_scale='Viridis'
    )
    return fig


def show_ef_ft_port(pvols, prets, tvols, trets, opts, optv, statistics):
    # Create a scatter plot for random portfolio compositions
    fig = go.Figure()

    # Scatter plot for random portfolios
    fig.add_trace(go.Scatter(
        x=pvols, 
        y=prets, 
        mode='markers', 
        marker=dict(
            color=prets / pvols,
            colorscale='Viridis',
            colorbar=dict(title='Sharpe Ratio'),
            size=8,
            opacity=0.6
        ),
        name='Random Portfolios'
    ))

    # Scatter plot for efficient frontier
    fig.add_trace(go.Scatter(
        x=tvols,
        y=trets,
        mode='markers',
        marker=dict(
            color=trets / tvols,
            colorscale='Viridis',
            size=8,
            symbol='x',
            opacity=0.6
        ),
        name='Efficient Frontier'
    ))

    # Point for the portfolio with the highest Sharpe ratio
    fig.add_trace(go.Scatter(
        x=[statistics(opts['x'])[1]],
        y=[statistics(opts['x'])[0]],
        mode='markers',
        marker=dict(
            color='blue',
            size=12
        ),
        name='Highest Sharpe Ratio'
    ))

    # Point for the minimum variance portfolio
    fig.add_trace(go.Scatter(
        x=[statistics(optv['x'])[1]],
        y=[statistics(optv['x'])[0]],
        mode='markers',
        marker=dict(
            color='green',
            size=12
        ),
        name='Minimum Variance'
    ))

    # Update layout
    fig.update_layout(
        title='Portfolios located on Simulation',
        xaxis_title='Expected Volatility',
        yaxis_title='Expected Return',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=800,
        height=400
    )

    return fig