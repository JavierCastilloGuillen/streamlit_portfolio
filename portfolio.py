import pandas as pd
import numpy as np
import streamlit as st

from helpers import  current_year, get_data, create_returns_plot, create_chart, create_ef_ft, show_ef_ft_port, stock_input
from optimization import run_optimization


option = st.sidebar.selectbox('Sections', ('test1','Portfolio Optimization','test3','Information and Disclaimer'))

st.write("""
    # Portfolio Optimization Tool
    In this example, we will apply the ***Efficient Frontier*** implementation using MonteCarlo Simulations from the [Modern Portfolio Theory (MPT)](https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/modern-portfolio-theory-mpt/) to define and optimize 2 portfolio examples.
    One by *reducing volatility* and the other by getting *optimal Sharpe Ratio*. 
    ***
""")

def main():
    
    if 'run_button_clicked' not in st.session_state:
        st.session_state.run_button_clicked = False

    # Sidebar for year selection
    st.sidebar.header('Metrics and Stocks')
    startyear = st.sidebar.selectbox('Consider stocks from', list(reversed(range(2012, current_year + 1)))) 

    # Sidebar for stock selection

    stock_input()

    # Add a GO button
    run_button = st.sidebar.button('GO')

    if run_button:
            st.session_state.run_button_clicked = True

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


    if st.session_state.run_button_clicked:
        noa = len(st.session_state.stocks)
        weights = np.ones(noa) / noa  # Equal weights for simplicity

        # Retrieve data
        data = get_data(st.session_state.stocks, startyear)
        if data.empty:
            st.error("No valid data available for the selected stocks. Please check your inputs.")
            st.session_state.stocks = []
            return

    if option == 'test1':
        st.title(st.session_state)

        if not st.session_state.run_button_clicked:
            st.info("ℹ️ Please confirm your stocks first. Select and Go")

        else:
            pass
            # Logic for 1 here



    if option == 'Portfolio Optimization':
        if not st.session_state.run_button_clicked:
            st.info("ℹ️ Please confirm your stocks first. Select and Go")
        
        else:
            run_optimization(data, weights, noa)


    if option == 'test3':
        st.title(st.session_state)

        if not st.session_state.run_button_clicked:
            st.info("ℹ️ Please confirm your stocks first. Select and Go")

        else:
            pass
            # Logic for 1 here


    if option == 'test4':

        if not st.session_state.run_button_clicked:
            st.info("ℹ️ Please confirm your stocks first. Select and Go")

        else:
            pass
            symbols_array = ','.join([f'["{symbol}", "{symbol}|1D"]' for symbol in st.session_state.stocks])
            tradingview_widget = """
                                <!-- TradingView Widget BEGIN -->
                                    <div class="tradingview-widget-container" style="height:100%;width:100%">
                                    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                                    {
                                    "autosize": true,
                                    "symbol": "NASDAQ:AAPL",
                                    "interval": "D",
                                    "timezone": "Etc/UTC",
                                    "theme": "dark",
                                    "style": "1",
                                    "locale": "en",
                                    "hide_top_toolbar": true,
                                    "allow_symbol_change": false,
                                    "calendar": false,
                                    "hide_volume": true,
                                    "support_host": "https://www.tradingview.com"
                                    }
                                    </script>
                                    </div>
                                    <!-- TradingView Widget END -->
                                """
            from streamlit import components
            st.components.v1.html(tradingview_widget, height=600)


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






# footer="""<style>
#             a:link , a:visited{
#             color: blue;
#             background-color: transparent;
#             text-decoration: underline;
#             }

#             a:hover,  a:active {
#             color: red;
#             background-color: transparent;
#             text-decoration: underline;
#             }

#             .footer {
#             position: fixed;
#             left: 0;
#             bottom: 0;
#             width: 100%;
#             background-color: white;
#             color: black;
#             text-align: center;
#             }
#             </style>
#             <div class="footer">
#             <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Heflin Stephen Raj S</a></p>
#             </div>
#             """
# st.markdown(footer,unsafe_allow_html=True)
                

if __name__ == "__main__":
    main()