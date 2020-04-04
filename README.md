# Factor-risk-model-with-principal-component-analysis
<p align="justify">In this repository, principal component analysis (PCA) is used on price time series of assets/securities to give an underlying composition into those of some selected risk factors price timeseries. In this example, we use different indexes tracking the NASDAQ, EUROSTOXX50, HSI or different bond markets to track a global market of different securities. Their price timeseries define a statistical risk model.</p>

<p align="justify">To elaborate further, the use of the PCA technique enables the formulation of statistical model factors (principal explanatory component, in this case) by clustering securities in sets in order to maximize asset return correlation within the cluster. At the same time, the clustered securities will have negligible correlations with the rest of the securities’ returns, thus enabling the derived factors to capture maximum risk.</p>

## Getting Started

<p align="justify">These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.</p>

### Prerequisites

<p align="justify">You need Python 3.x to run the following code.  You can have multiple Python versions (2.x and 3.x) installed on the same system without problems. Python needs to be first installed then SciPy</p>.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    sudo apt-get install python3 python3-pip

Alongside Python, the SciPy packages are also required. In Ubuntu and Debian, the SciPy ecosystem can be installed by:

    sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

For getting the data of price timeseries, we use the yfinance package that can be installed using the <a href="https://pypi.org/project/pip/">pip</a> package manager:

    pip install yfinance

For more information on the packages and their installation process, check the below links:

http://www.python.org/getit/ for Python    
https://www.scipy.org/install.html for the SciPy ecosystem    
https://pypi.org/project/yfinance/ for the yfinance module 


### File descriptions
<ul>
  
<li><div align="justify">'<em>Backtester_loop.py</em>' in which the Backtest class hierarchy encapsulates the other classes, to carry out a nested while-loop event-driven system in order to handle the events placed on the Event Queue object.</div></li>
    
<li><div align="justify">'<em>DataHandler.py</em>' which defines a class that gives all subclasses an interface for providing market data to the remaining components within the system. Data can be obtained directly from the web, a database or be read from CSV files for instance.</div></li>

<li><div align="justify">'<em>Events.py</em>' with four types of events (market, signal, order and fill), which allow communication between the above components via an event queue, are implemented.</div></li>

<li><div align="justify">'<em>Execution.py</em>' to simulate the order handling mechanism and ultimately tie into a brokerage or other
means of market connectivity.</div</li>

<li><div align="justify">'<em>Main.py</em>' which is the main Python program, englobing all the different subroutines, and where the different parameters to initialize the backtesting simulations are specified.</div</li>

<li><div align="justify">'<em>Performance.py</em>' in which performance assessment criteria are implemented such as the Sharpe ratio and drawdowns.</div</li>
  
<li><div align="justify">'<em>Plot_Performance.py</em>' to plot figures based on the equity curve obtained after backtesting.</div</li>
  
<li><div align="justify">'<em>Portfolio.py</em>' that keeps track of the positions within a portfolio, and generates orders of a fixed quantity of stock based on signals.</div></li>

<li><div align="justify">'<em>Strategy.py</em>' to generate a signal event from a particular strategy to communicate to the portfolio.</div></li>

<li><div align="justify">In the '<em>Strategies</em>' directory, different trading strategies are implemented to be used for backtesting:</div></li>

  <ul>
    <li><div align="justify">'<em>Buy_and_hold_strat.py</em>' in which a simple buy and hold strategy is coded.</div></li>
  <li><div align="justify">'<em>Moving_average_crossover_strat.py</em>' to generate signals from simple moving averages.</div></li>
  <li><div align="justify">'<em>ETF_forecast.py</em>' basic forecasting algorithm on ETF such as S&P500 using lagged price data</div></li>
  </ul>
  
</ul>

### Running the program

The different "<em>.py</em>" files need to be placed in the same folder for the main script to be run. The code is then ready to be used, and just requires running the following command:

    python Main.py

<p align="justify">The code is well commented and easy to understand. The different parameters calculated and used for the current mappings are:</p>

``` python
factors_symbols = ['^NDX', '^STOXX50E', '^HSI', 'IEF', 'SHY', 'VIXY', 'LQD', 'HYG', 'IBND', 'TIP']
securities_symbols = ['AAPL', 'MSFT', 'AMZN', 'BRK-A', 'AC.PA', 'AIR.PA', 'DAL', 'MAR', 'DIS', 'SIX', 'VOW3.DE',
                      '0700.HK', '2318.HK', 'BABA', '0005.HK', '1299.HK', 'ENI.MI', 'ALV.DE']

start_date = "2015-01-01"
end_date = "2020-01-01"
read_csv = True
```

<p align="justify">By running the main script, the resulting csv file '<em>pcaMappingsResults.csv</em>' is generated, containing the mappings coefficients of securities to risk factors. Below is a snapshot of the results.</p>

<p align="justify">It can be seen that HK securities are mainly explained by the HSI index, or big US firms such as AMZN of GOOG have a coefficient near 1 for the Nasdaq index ('^NDX') </p>

<p align="center">
<img src="https://github.com/DavidCico/Factor-risk-model-with-principal-component-analysis/blob/master/mappings_results.JPG" width="1500" height="330">
</p>

## Contributing

Please read [CONTRIBUTING.md](https://github.com/DavidCico/Factor-risk-model-with-principal-component-analysis/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

## Authors

* **David Cicoria** - *Initial work* - [DavidCico](https://github.com/DavidCico)

See also the list of [contributors](https://github.com/DavidCico/Factor-risk-model-with-principal-component-analysis/graphs/contributors) who participated in this project.
