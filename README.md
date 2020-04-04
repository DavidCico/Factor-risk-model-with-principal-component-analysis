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
    <li>'<em>ETF_data</em>' which is a univariate time series of the price history of the ETF.</li>
    <li>'<em>Main.py</em>' which contains the main procedure, as well as the data pre-processing of the xlsx file 'ETF_data.xlsx'</li>
    <li>'<em>Monte_Carlo_GBM.py</em>' which contains the different algorithms used for comparison.</li>
<li><div align="justify">'<em>Post_processing.py</em>' where all the functions for post-processing (plots, information, descriptive statistics) are implemented.</div></li>
<li><div align="justify">'<em>Analysis.pdf</em>', the PDF file where the different steps of the financial study are explained.</div></li>
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

<p align="justify">By running the main script, the resulting csv file '<em>pcaMappingsResults.csv</em>' is generated, containing the mappings coefficients of securities to risk factors.</p>

<p align="center">
<img src="https://github.com/DavidCico/Study-of-buy-and-hold-investment/blob/master/Example_Results/analytic_exp_gbm.png" width="500" height="350"> <img src="https://github.com/DavidCico/Study-of-buy-and-hold-investment/blob/master/Example_Results/Hists_fig2.jpg" width="512" height="512" >
</p>

## Contributing

Please read [CONTRIBUTING.md](https://github.com/DavidCico/Factor-risk-model-with-principal-component-analysis/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

## Authors

* **David Cicoria** - *Initial work* - [DavidCico](https://github.com/DavidCico)

See also the list of [contributors](https://github.com/DavidCico/Factor-risk-model-with-principal-component-analysis/graphs/contributors) who participated in this project.
