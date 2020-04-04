# Factor-risk-model-with-principal-component-analysis
<p align="justify">In this repository, principal component analysis (PCA) is used on price time series of assets/securities to give an underlying composition into those of some selected risk factors price timeseries. In this example, we use different indexes tracking the NASDAQ, EUROSTOXX50, HSI or different bond markets to track a global market of different securities. Their price timeseries define a statistical risk model.</p>

<p align="justify">To elaborate further, the use of the PCA technique enables the formulation of statistical model factors (principal explanatory component, in this case) by clustering securities in sets in order to maximize asset return correlation within the cluster. At the same time, the clustered securities will have negligible correlations with the rest of the securities’ returns, thus enabling the derived factors to capture maximum risk.</p>

<p align="justify">The price of the different securities selected (or from an asset universe in a real case) can then be defined as a linear combination of factor timeseries. As the number of factors is usually small compared to an universe of assets, realizing calculations on only the different factors, such as optimization, recalculation of correlation matrix on daily basis, are improved in terms of computational efficiency as well as getting the ability to plot data in an easier manner. This is usually the main interest of PCA when applied in different fields.</p>

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
  
<li><div align="justify">'<em>data_feed.py</em>' in which the data query from Yahoo finance and dataframe manipulations functions are located.</div></li>
    
<li><div align="justify">'<em>main.py</em>' which is the main Python program, englobing all the different subroutines, and where the different parameters to initialize the pca mappings are specified. </div></li>

<li><div align="justify">'<em>mappings.py</em>' contains the functions of mapping securities to risk factors</div></li>.

<li><div align="justify">'<em>pcaMappingsResults.csv</em>', the csv file obtained as the result of running the code. It contains the mapping coefficients of securities to risk factors, as well as some information such as 'pct_systematic_ratio' (ratio of variance explained by the PCA), correlation to PCA or mapped VaR</div</li>.

<li><div align="justify">'<em>portfolio_decomposition.py</em>' is where the main content of the program is located. The different functions in this file, consist of the generation of PCA mappings for the risk factor series, and the linear regression of securities prices to PCA prices, in order to generate the final mapping of securities to factors.</div</li>

<li><div align="justify">'<em>securities_and_factors.csv</em>' to be read by the program depending on the user's choice (get data from Yahoo or csv file).</div</li>

<li><div align="justify">'<em>var_exp_weighted.py</em>' in which exponentially weighted VaR calculation is implemented.</div</li>
  
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

<p align="justify">It can be seen that HK securities are mainly explained by the HSI index, or big US firms such as AMZN of GOOG have a big coefficient of regression with the Nasdaq index ('^NDX'), with some 0.2 components with high yield corporate bonds ('HYG'). The price timeseries of the different securities can then be defined as a linear combination of the different risk factors, with the weight of the components showing their underlying composition compared to these.</p>

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
