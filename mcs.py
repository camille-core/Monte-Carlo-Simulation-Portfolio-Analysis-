import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import read

#Data preparation
stocks = pd.read_csv('stocks.csv')
stocks.rename(columns={'Change%':'Change'}, inplace=True)
stocks = stocks[stocks['Code'].isin(['AC', 'BDO', 'JGS', 'SM', 'AP'])]
stocks['Change'] = stocks['Change'].replace('%', '', regex=True).astype(float) / 100
stocks['Date'] = pd.to_datetime(stocks['Date'], format="%b %d, %Y")
stocks = stocks.sort_values(['Code', 'Date'])
stocks = stocks.loc[stocks['Date']>='2019-12-24']
print(stocks)

meanReturns = stocks.groupby('Code')['Change'].mean()
#print(meanReturns)
grouped = stocks.pivot(index='Date', columns='Code', values='Change')
#print(grouped)
covMatrix = grouped.cov()
#print(covMatrix)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

#MCS: With help from QuantPy
sims = 70
T=(stocks['Date'].max()-stocks['Date'].min()).days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
print(meanM)

portfolio_sims = np.full(shape=(T, sims), fill_value=0.0)

inPortfolio = 100000

for m in range(0, sims):
    #MCS loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*inPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value (PHP)')
plt.xlabel('Days')
plt.title('MC Simulation of a PSE Stock Portfolio')
plt.show()


