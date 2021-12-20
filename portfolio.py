import numpy as np
import datetime as dt
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import scipy.optimize as sco
import streamlit as st
import seaborn as sns
import plotly.express as px
sns.set_theme()

def create_chart(stocks,weights):
    labels = stocks
    sizes = weights
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    return fig1

@st.cache(suppress_st_warning=True)
def get_data(symbols):
    startmonth = 1 
    startday = 1
    start = dt.datetime(startyear, startmonth, startday)
    now = dt.datetime.now()
    data = pdr.get_data_yahoo(symbols, start , now)
    data = data['Adj Close']
    return data

def create_ef_ft(pvols, prets):
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    plt.grid(True)
    plt.title('MonteCarlo Simulation Porfolios')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    return fig

def show_ef_ft_port(pvols,prets,tvols,trets):
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    # random portfolio composition
    plt.scatter(tvols, trets, c=trets / tvols, marker='x')
    # efficient frontier
    plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'bo', markersize=10.0)
    # portfolio with highest Sharpe ratio
    plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'go', markersize=10.0)
    # minimum variance portfolio
    plt.grid(True)
    plt.title('Portfolios located on Simulation')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    
    return fig

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




# Sidebar
st.sidebar.header('Select starting year')
startyear = st.sidebar.selectbox('Year', list(reversed(range(2012,2021))))
st.sidebar.header('Stock selection to optimize')

stock_1 = st.sidebar.text_input("Asset 1", value='AAPL'.upper())
stock_2 = st.sidebar.text_input("Asset 2", value='F'.upper())
stock_3 = st.sidebar.text_input("Asset 3", value='MSFT'.upper())
stock_4= st.sidebar.text_input("Asset 4", value='JPM'.upper())

weights = np.array([0.25,0.25,0.25,0.25,])

st.write("""
    # Porfolio Optimization Tool
    On this example we will apply the ***Efficient Frontier*** implementation using MonteCarlo Simulations from the [Modern Portfolio Theory (MPT)](https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/modern-portfolio-theory-mpt/) to define and optimize 2 portfolio examples.
    One by *reducing volatility* and other by getting *optimal Sharpe Ratio*. 
    ***
    """)

expander = st.expander('How does it work?')
expander.markdown("""
    * Define a portfolio of ***4 assets*** on the sidebar and select the start date for the data retreival.
    * Implement a **MonteCarlo Simulation** (limited to 10000 due computational efficiency for the example) to [get the Efficient Frontier.](https://en.wikipedia.org/wiki/Efficient_frontier) 
    * We will get the metrics and weights for an ***Optimal Sharpe Portfolio*** and a ***Minimum Variance Portfolio*** *(less volatility)*.
    * Notice the optimal portfolios might have less than the inital assets introduced!
    * For the example data is gathered using Yahoo! Finance. Use that ticker format. Ex: S&P500 = [^GSPC](https://finance.yahoo.com/quote/%5EGSPC/) or [YPFD.BA](https://finance.yahoo.com/quote/YPFD.BA/), [BBVA.MC](https://es.finance.yahoo.com/quote/bbva.mc?ltr=1) for local markets.
    """)

symbols = [stock_1, stock_2, stock_3, stock_4]
noa = len(symbols)

data = get_data(symbols)

# Essential Metrics
rets = np.log(data / data.shift(1))
returns_plot = rets.cumsum().apply(np.exp)
s_ret = data.pct_change()
portfolio_simple_returns = np.sum(s_ret.mean() * weights) * 252
rets_an_mean = ((rets.mean() * 252) * 100)
rets_cov = rets.cov() * 252
port_variance = np.dot(weights.T, np.dot(rets.cov() * 252, weights))
port_volatility = np.sqrt(port_variance)

d_eq = {'Volatility [%]': port_volatility *100, 'Simple Return [%]': portfolio_simple_returns *100}
df_eq = pd.DataFrame(d_eq, index=[0])

# Simulations
prets = []
pvols = []

for p in range (10**4):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)

st.write("""
    ### 1. Asset metrics
    This is the performance of your selection, from the selected date. A lot of metrics can be extracted from the stock selection. 
    Do you think it's a good selection? Does this asset selection match your ***risk profile***? How they would have perform from your date selection?
    """)

st.table(df_eq)
fig = px.line(returns_plot, title='Simple Returns Plot')
st.plotly_chart(fig, use_container_width=True)

st.write("""
    ### 2. Your MonteCarlo Simulation
    On this chart, you'll have the ***Expected Volatility***, ***Expected Return** and ***Sharpe Ratio*** for the simulated weight distribution.
    Each single dot, correspond to a ***different weight allocation portfolio***.
    However, from all of them, we want to obtain those portfolios with 2 characteristics on this example:
    The less volatile for the higher returns *(Green Dot on chart below)*, and the one with the optimal Sharpe Ratio *(Blue Dot on chart below)*.
    """)

ef_ft = create_ef_ft(pvols, prets)
st.pyplot(ef_ft)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))

opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
results_sharpe = opts['x'].round(3)
values_sharpe = statistics(opts['x']).round(3)

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
results_var = optv['x'].round(3)
values_var = statistics(optv['x']).round(3)

trets = np.linspace(prets.min(), prets.max(), 100)
tvols = []

for tret in trets:
   cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
   res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                        bounds=bnds, constraints=cons)
   tvols.append(res['fun'])
tvols = np.array(tvols)

stars = show_ef_ft_port(pvols,prets,tvols,trets)

## Dataframes to plot results
d_s = {'Return': values_sharpe[0], 'Volatility': values_sharpe[1], 'Sharpe Ratio':values_sharpe[2]}
df_s = pd.DataFrame(d_s, index=[0])
d_sa = {stock_1: results_sharpe[0], stock_2: results_sharpe[1],stock_3: results_sharpe[2],stock_4 : results_sharpe[3]}
df_sa = pd.DataFrame(d_sa, index=[0])

d_v = {'Return': values_var[0], 'Volatility': values_var[1], 'Sharpe Ratio':values_var[2]}
df_v = pd.DataFrame(d_v, index=[0])
d_va = {stock_1: results_var[0], stock_2: results_var[1],stock_3: results_var[2],stock_4 : results_var[3]}
df_va = pd.DataFrame(d_va, index=[0])


st.write("""
    ### 3. Optimal Sharpe Portfolio
    ***Sharpe ratio:*** One of the most popular performance evaluation metrics, it
    measures the excess return (over the risk-free rate) per unit of standard
    deviation. When no risk-free rate is provided, the default assumption is that it is
    equal to 0%. The greater the Sharpe ratio, the better the portfolio's risk-adjusted
    performance.
    """)
st.table(df_s)
st.write("""
    ##### Weight Distribution
    """)
st.table(df_sa)
sharpe_chart = create_chart(symbols,results_sharpe)
st.pyplot(sharpe_chart)

st.write("""
    ### 4. Minimum Variance Portfolio
    The ***Minimum Variance*** Portfolio could be described as well as the less volatile portfolio.
    The weight distribution below corresponds to this metrics.
    """)
st.table(df_v)
st.write("""
    ##### Weight Distribution
    """)
st.table(df_va)

variance_chart = create_chart(symbols,results_var)
st.pyplot(variance_chart)


st.write("""
    ### 5. Where this portfolios are located?
    As expected, the portfolios are located over the ***Efficient Frontier***. 
    The ***efficient frontier*** is comprised of all optimal portfolios with a higher return than the
    absolute minimum variance portfolio. These portfolios dominate all other portfolios in
    terms of expected returns given a certain risk level.
    
    ***Green Dot:*** Minimum Variance Portfolio (Less volatile for higher return).
    ***Blue Dot:*** Optimal Sharpe Ratio (Higer return for less volatility)
    Portfolios with different metrics can be obtained, however, for this open example we will leave it here.
    """)
st.pyplot(stars)

st.write("""
    ### Final Considerations
    
    Further analysis can be continued from here. For this public script we will stop here for now. 
    Repository will be available on [GitHub](https://github.com/JavierCastilloGuillen).
    """)

expander = st.expander('Notes / Information / Contact')
expander.markdown("""
    * Note this is a public example, some capabilities are limited to simplify the app. If you have a doubt or you wish any other usage, get in touch.
    * Limitations: Number of iteration for MCS, number of assets, dates, data source, metrics to get specific porftolios other than Sharpe and Volatility, etc. 
    * If you've got any feedback or comment, I'll be happy to read it ;). 
    * For this examples, ideas and more contact [here.](https://jcgmarkets.com/en/javier-castillo/)
    """)

st.write("""
    #### References:
    1. Cf. Markowitz, Harry (1952): “Portfolio Selection.” Journal of Finance, Vol. 7, 77-91.
    2. Hilpisch, Yves (2015): “Python For Finance. Analyze Big Financial Data”.
    3. Rothwell, Kevin (2020): “Applied Financial Advice and Wealth Management”
    4. [Streamlit documentation](https://docs.streamlit.io/library/api-reference)
    """)
