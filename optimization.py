import streamlit as st
import numpy as np
import pandas as pd
from helpers import create_returns_plot, create_ef_ft, create_chart, show_ef_ft_port
import scipy.optimize as sco


def run_optimization(data, weights, noa):
    if not st.session_state.run_button_clicked:
        st.info("ℹ️ Please confirm your stocks first. Select and Go")

    else:
        st.write(st.session_state)    
        

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

        with st.spinner("Processing... Please wait."):

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
            d_sa = {st.session_state.stocks_opt[i]: results_sharpe[i] for i in range(noa)}
            df_sa = pd.DataFrame(d_sa, index=[0])

            d_v = {'Return': values_var[0], 'Volatility': values_var[1], 'Sharpe Ratio': values_var[2]}
            df_v = pd.DataFrame(d_v, index=[0])
            d_va = {st.session_state.stocks_opt[i]: results_var[i] for i in range(noa)}
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
                ##### Weight Distribution in Portfolio in [%]
            """)
            st.dataframe(df_sa.round(2))
            sharpe_chart = create_chart(st.session_state.stocks_opt, results_sharpe)
            st.plotly_chart(sharpe_chart)

            st.write("""
                ### 4. Minimum Variance Portfolio
                The ***Minimum Variance*** Portfolio could be described as well as the less volatile portfolio.
                The weight distribution below corresponds to this metric.
            """)
            st.dataframe(df_v.round(2))
            st.write("""
                ##### Weight Distribution in Portfolio in [%]
            """)
            st.dataframe(df_va.round(2))

            variance_chart = create_chart(st.session_state.stocks_opt, results_var)
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
        with st.spinner("Processing... Please wait."):

            stars = show_ef_ft_port(pvols, prets, tvols, trets, opts, optv, statistics)
            st.plotly_chart(stars)
    
        st.session_state.stocks_opt = []