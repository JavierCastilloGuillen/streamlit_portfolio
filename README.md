# Porfolio Optimization Tool
On this example we will apply the ***Efficient Frontier*** implementation using MonteCarlo Simulations from the [Modern Portfolio Theory (MPT)](https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/modern-portfolio-theory-mpt/) to define and optimize 2 portfolio examples.
One by *reducing volatility* and other by getting *optimal Sharpe Ratio*. 
***

## Try it live

Try it [here.](https://share.streamlit.io/javiercastilloguillen/streamlit_portfolio/main/portfolio.py)

## How does it work?

* Define a portfolio of ***4 assets*** on the sidebar and select the start date for the data retreival.
* Implement a **MonteCarlo Simulation** (limited to 10000 due computational efficiency for the example) to [get the Efficient Frontier.](https://en.wikipedia.org/wiki/Efficient_frontier) 
* We will get the metrics and weights for an ***Optimal Sharpe Portfolio*** and a ***Minimum Variance Portfolio*** *(less volatility)*.
* Notice the optimal portfolios might have less than the inital assets introduced!
* For the example data is gathered using Yahoo! Finance. Use that ticker format. Ex: S&P500 = [^GSPC](https://finance.yahoo.com/quote/%5EGSPC/) or [YPFD.BA](https://finance.yahoo.com/quote/YPFD.BA/), [BBVA.MC](https://es.finance.yahoo.com/quote/bbva.mc?ltr=1) for local markets.


#### Notes / Information / Contact
* Note this is a public example, some capabilities are limited to simplify the app. If you have a doubt or you wish any other usage, get in touch.
* Limitations: Number of iteration for MCS, number of assets, dates, data source, metrics to get specific porftolios other than Sharpe and Volatility, etc. 
* If you've got any feedback or comment, I'll be happy to read it ;). 
* For this examples, ideas and more contact [here.](https://jcgmarkets.com/en/javier-castillo/)

#### References:
1. Cf. Markowitz, Harry (1952): “Portfolio Selection.” Journal of Finance, Vol. 7, 77-91.
2. Hilpisch, Yves (2015): “Python For Finance. Analyze Big Financial Data”.
3. Rothwell, Kevin (2020): “Applied Financial Advice and Wealth Management”
4. [Streamlit documentation](https://docs.streamlit.io/library/api-reference)
