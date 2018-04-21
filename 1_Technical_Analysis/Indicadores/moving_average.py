# Moving Average Variations	

import numpy as np
import matplotlib.pyplot as plt
import datetime

# Simple Moving Average
def SMA(values, window):
	weights = np.repeat(1.0, window)/window
	sma = np.convolve(values, weights, 'valid')
	return sma

# Exponential Moving Average
def EMA(values, window):
	weights = np.exp(np.linspace(-1., 0., window))
	weights /= weights.sum()
	ema = np.convolve(values, weights, mode = 'full')[:len(values)]
	ema[:window] = ema[window]
	return ema

# Weighted Moving Average
def WMA(values, window):
	weights = np.arange(1, len(values)+1)
	wma = np.zeros(len(values)-window+1)
	endset = window
	for k in range(0, len(values)):
		subset = values[k:endset]
		subweights = weights[k:endset]
		wa = subset * subweights
		wma[k] = wa.sum() / weights.sum()
		endset = endset + 1
		if k+window == len(values):
			break
	return wma

# Random Samples
dataset = np.random.randint(10, size=(20))

# Window Size
window = 3

# Time Series Plots
plt.plot(SMA(dataset, window), label='SMA')
print('SMA', SMA(dataset, window))
plt.plot(EMA(dataset, window), label='EMA')
print('EMA', EMA(dataset, window))
plt.plot(WMA(dataset, window), label ='WMA')
print('WMA', WMA(dataset, window))
plt.legend()
plt.show()