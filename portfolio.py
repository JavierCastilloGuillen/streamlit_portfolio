import pandas as pd
import numpy as np
import streamlit as st
import scipy.optimize as sco
import seaborn as sns
from streamlit_tags import st_tags_sidebar
import plotly.graph_objects as go
from helpers import  current_year, get_data, create_returns_plot, create_chart, create_ef_ft, show_ef_ft_port
sns.set_theme()

st.write("""
    # Portfolio Optimization Tool
    In this example, we will apply the ***Efficient Frontier*** implementation using MonteCarlo Simulations from the [Modern Portfolio Theory (MPT)](https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/modern-portfolio-theory-mpt/) to define and optimize 2 portfolio examples.
    One by *reducing volatility* and the other by getting *optimal Sharpe Ratio*. 
    ***
""")

def main():


    def min_func_port(weights):
        return statistics(weights)[1]

    def statistics(weights):
        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])

    def min_func_variance(weights):
        return statistics(weights)[1] ** 2

    def min_func_sharpe(weights):
        return -statistics(weights)[2]



    # Sidebar for year selection
    st.sidebar.header('Select starting year')
    startyear = st.sidebar.selectbox('Year', list(reversed(range(2012, current_year + 1)))) 

    # Sidebar for stock selection
    # Initialize session state for stock tickers
    if 'stocks' not in st.session_state:
        st.session_state.stocks = ['AAPL', 'F', 'MSFT', 'JPM']


    # Initialize session state for stock tickers
    if 'stocks' not in st.session_state:
        st.session_state.stocks = ['AAPL', 'F', 'MSFT', 'JPM']

    # Ticker selection using st_tags
    symbols = st_tags_sidebar(
        label='Tickers on Portfolio',
        text='Press enter to add more',
        value=st.session_state.stocks,
        suggestions=['AAPL', 'F', 'MSFT', 'JPM'],
        maxtags=20,
        key='1'
    )
    if len(symbols) == 0:
        st.sidebar.warning("⚠️ Please select your stocks first.")


    # Convert all symbols to uppercase
    symbols = [symbol.upper() for symbol in symbols]

    # Update session state with current symbols
    st.session_state.stocks = symbols
    # Add a GO button
    run_button = st.sidebar.button('GO')
    
    
    st.sidebar.markdown("")

    st.sidebar.markdown("")

    st.sidebar.markdown("---")

    with st.sidebar:
        with st.popover("How does it work?"):
            st.markdown("""
            * Define a portfolio of ***multiple assets*** on the sidebar and select the start date for the data retrieval.
            * Implement a **MonteCarlo Simulation** (limited to 10000 due to computational efficiency for the example) to [get the Efficient Frontier.](https://en.wikipedia.org/wiki/Efficient_frontier) 
            * We will get the metrics and weights for an ***Optimal Sharpe Portfolio*** and a ***Minimum Variance Portfolio*** *(less volatility)*.
            * Notice the optimal portfolios might have less than the initial assets introduced!
            * For the example, data is gathered using Yahoo! Finance. Use that ticker format. Ex: S&P500 = [^GSPC](https://finance.yahoo.com/quote/%5EGSPC/) or [YPFD.BA](https://finance.yahoo.com/quote/YPFD.BA/), [BBVA.MC](https://es.finance.yahoo.com/quote/bbva.mc?ltr=1) for local markets.
            """)

    if run_button:
        noa = len(symbols)
        weights = np.ones(noa) / noa  # Equal weights for simplicity

        st.write(f"Selected start year: {startyear}")
        st.write(f"Selected stocks: {symbols}")
        st.write(f"Weights: {weights}")

        # Portfolio Optimization Tool Description

        # Retrieve data
        data = get_data(symbols, startyear)
        if data.empty:
            st.error("No valid data available for the selected stocks. Please check your inputs.")
            st.session_state.stocks = []
            return


        # Essential Metrics
        rets = np.log(data / data.shift(1))

        returns_plot = rets.cumsum().apply(np.exp)
        s_ret = data.pct_change()
        portfolio_simple_returns = np.sum(s_ret.mean() * weights) * 252
        rets_an_mean = ((rets.mean() * 252) * 100)
        rets_cov = rets.cov() * 252
        port_variance = np.dot(weights.T, np.dot(rets.cov() * 252, weights))
        port_volatility = np.sqrt(port_variance)

        d_eq = {'Volatility [%]': (port_volatility * 100), 'Simple Return [%]': portfolio_simple_returns * 100}
        df_eq = pd.DataFrame(d_eq, index=[0])

        # Simulations
        prets = []
        pvols = []
        num_portfolios = 10**4 ###############################################################################################################
        with st.spinner("Processing... Please wait."):
            # Generate random weights
            weights = np.random.random((num_portfolios, noa))
            weights /= np.sum(weights, axis=1)[:, np.newaxis]

            # Calculate portfolio returns and volatilities
            prets = np.sum(weights * rets.mean().values, axis=1) * 252
            pvols = np.sqrt(np.einsum('ij,jk,ik->i', weights, rets.cov().values * 252, weights))

            # Convert to numpy arrays (if needed)
            prets = np.array(prets)
            pvols = np.array(pvols)

        st.write("""
            ### 1. Asset metrics
            This is the performance of your selection, from the selected date. A lot of metrics can be extracted from the stock selection. 
            Do you think it's a good selection? Does this asset selection match your ***risk profile***? How they would have performed from your date selection?
        """)

        st.dataframe(df_eq.round(2))
        fig = create_returns_plot(returns_plot)
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
            ### 2. Your MonteCarlo Simulation
            On this chart, you'll have the ***Expected Volatility***, ***Expected Return***, and ***Sharpe Ratio*** for the simulated weight distribution.
            Each single dot corresponds to a ***different weight allocation portfolio***.
            However, from all of them, we want to obtain those portfolios with 2 characteristics on this example:
            The less volatile for the higher returns *(Green Dot on chart below)*, and the one with the optimal Sharpe Ratio *(Blue Dot on chart below)*.
        """)

        ef_ft = create_ef_ft(pvols, prets)
        st.plotly_chart(ef_ft)

        # Define constraints and bounds
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(noa))

        # Optimization for Sharpe ratio
        opts = sco.minimize(min_func_sharpe, noa * [1. / noa], method='SLSQP', bounds=bnds, constraints=cons)
        results_sharpe = opts['x'].round(3)
        values_sharpe = statistics(opts['x']).round(3)

        # Optimization for minimum variance
        optv = sco.minimize(min_func_variance, noa * [1. / noa], method='SLSQP', bounds=bnds, constraints=cons)
        results_var = optv['x'].round(3)
        values_var = statistics(optv['x']).round(3)

        # Efficient frontier
        trets = np.linspace(prets.min(), prets.max(), 100)

        # Optimize portfolio for each target return
        tvols = np.array([
            sco.minimize(
                min_func_port,
                noa * [1. / noa],
                method='SLSQP',
                bounds=bnds,
                constraints=[
                    {'type': 'eq', 'fun': lambda x, tret=tret: statistics(x)[0] - tret},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                ]
            )['fun'] for tret in trets
        ])


        # Dataframes to plot results
        d_s = {'Return': values_sharpe[0], 'Volatility': values_sharpe[1], 'Sharpe Ratio': values_sharpe[2]}
        df_s = pd.DataFrame(d_s, index=[0])
        d_sa = {symbols[i]: results_sharpe[i] for i in range(noa)}
        df_sa = pd.DataFrame(d_sa, index=[0])

        d_v = {'Return': values_var[0], 'Volatility': values_var[1], 'Sharpe Ratio': values_var[2]}
        df_v = pd.DataFrame(d_v, index=[0])
        d_va = {symbols[i]: results_var[i] for i in range(noa)}
        df_va = pd.DataFrame(d_va, index=[0])

        st.write("""
            ### 3. Optimal Sharpe Portfolio
            ***Sharpe ratio:*** One of the most popular performance evaluation metrics, it
            measures the excess return (over the risk-free rate) per unit of standard
            deviation. When no risk-free rate is provided, the default assumption is that it is
            equal to 0%. The greater the Sharpe ratio, the better the portfolio's risk-adjusted
            performance.
        """)
        st.dataframe(df_s.round(2))
        st.write("""
            ##### Weight Distribution
        """)
        st.dataframe(df_sa.round(2))
        sharpe_chart = create_chart(symbols, results_sharpe)
        st.plotly_chart(sharpe_chart)

        st.write("""
            ### 4. Minimum Variance Portfolio
            The ***Minimum Variance*** Portfolio could be described as well as the less volatile portfolio.
            The weight distribution below corresponds to this metric.
        """)
        st.dataframe(df_v.round(2))
        st.write("""
            ##### Weight Distribution
        """)
        st.dataframe(df_va.round(2))

        variance_chart = create_chart(symbols, results_var)
        st.plotly_chart(variance_chart)

        st.write("""
            ### 5. Where these portfolios are located?
            As expected, the portfolios are located over the ***Efficient Frontier***. 
            The ***efficient frontier*** is comprised of all optimal portfolios with a higher return than the
            absolute minimum variance portfolio. These portfolios dominate all other portfolios in
            terms of expected returns given a certain risk level.
            
            ***Green Dot:*** Minimum Variance Portfolio (Less volatile for higher return).
            ***Blue Dot:*** Optimal Sharpe Ratio (Higher return for less volatility)
            Portfolios with different metrics can be obtained, however, for this open example we will leave it here.
        """)
        # Plot results
        stars = show_ef_ft_port(pvols, prets, tvols, trets, opts, optv, statistics)
        st.plotly_chart(stars)

        st.write("""
            ### Final Considerations
            
            Further analysis can be continued from here. For this public script, we will stop here for now. 
            Repository will be available on [GitHub](https://github.com/JavierCastilloGuillen).
        """)

        expander = st.expander('Notes / Information / Contact')
        expander.markdown("""
            * Note this is a public example, some capabilities are limited to simplify the app. If you have a doubt or you wish any other usage, get in touch.
            * Limitations: Number of iterations for MCS, number of assets, dates, data source, metrics to get specific portfolios other than Sharpe and Volatility, etc. 
            * If you've got any feedback or comment, I'll be happy to read it ;). 
            * For this examples, ideas and more contact [here.](https://javiercg.com/javier-castillo/)
        """)

        st.write("""
            #### References:
            1. Cf. Markowitz, Harry (1952): “Portfolio Selection.” Journal of Finance, Vol. 7, 77-91.
            2. Hilpisch, Yves (2015): “Python For Finance. Analyze Big Financial Data”.
            3. Rothwell, Kevin (2020): “Applied Financial Advice and Wealth Management”
            4. [Streamlit documentation](https://docs.streamlit.io/library/api-reference)
        """)
        st.session_state.stocks = []
          

if __name__ == "__main__":
    main()