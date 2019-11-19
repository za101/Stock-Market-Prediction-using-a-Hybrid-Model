import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from pandas import Series
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import math
import datetime
import warnings
import os, os.path
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf



def timeseries(company_name):
	df = pd.read_csv('data_files/WIKI-'+company_name+'.csv')
	print(len(df))
	df['Date'] = pd.to_datetime(df['Date'])
	df.set_index('Date', inplace = True)

	df = df[['Adj. Open',  'Adj. High',  'Adj. Low', 'Adj. Volume', 'Adj. Close']]
	df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
	df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
	
	df_timeseries_open = df[df.columns[0]]
	df_timeseries_high = df[df.columns[1]]
	df_timeseries_low= df[df.columns[2]]
	df_timeseries_vol = df[df.columns[3]]
	df_timeseries_close = df[df.columns[4]]
	df_timeseries_HL_PCT = df[df.columns[5]]
	df_timeseries_PCT_change = df[df.columns[6]]
	
	x1,train_size = timer(df_timeseries_open)
	print("done 1 ",train_size," ",len(x1))
	x2,train_size = timer(df_timeseries_high)
	print("done 2 ",train_size)
	x3,train_size = timer(df_timeseries_low)
	print("done 3 ",train_size)
	x4,train_size = timer(df_timeseries_vol)
	print("done 4 ",train_size)
	x6,train_size = timer(df_timeseries_HL_PCT)
	print("done 6 ",train_size)
	x7,train_size = timer(df_timeseries_PCT_change)
	print("done 7 ",train_size)
	"""
	np.savetxt('open.txt', x1, fmt='%d')
	np.savetxt('high.txt', x2, fmt='%d')
	np.savetxt('low.txt', x3, fmt='%d')
	np.savetxt('volume.txt', x4, fmt='%d')
	np.savetxt('Close.txt', x5, fmt='%d')
	np.savetxt('HL_PCT.txt', x6, fmt='%d')
	np.savetxt('PCT_change.txt', x7, fmt='%d')
	"""
	x1 = np.loadtxt('open.txt', dtype=int)
	x2 = np.loadtxt('high.txt', dtype=int)
	x3 = np.loadtxt('low.txt', dtype=int)
	x4 = np.loadtxt('volume.txt', dtype=int)
	x6 = np.loadtxt('HL_PCT.txt', dtype=int)
	x7 = np.loadtxt('PCT_change.txt', dtype=int)
	
	dfform = {'Adj. Open': df['Adj. Open'], 
        'Adj. High':df['Adj. High'], 
        'Adj. Low': df['Adj. Low'],
        'Adj. Volume': df['Adj. Volume'],
        'Adj. Close':df['Adj. Close'],
        'HL_PCT':df['HL_PCT'],
        'PCT_change':df['PCT_change']
    }
	
	df1 = pd.DataFrame(dfform) 
	
	data = {'Adj. Open': x1, 
        'Adj. High': x2, 
        'Adj. Low': x3,
        'Adj. Volume': x4,
        'HL_PCT': x6,
        'PCT_change': x7
    } 
    
	df_test = pd.DataFrame(data)

	y = np.array(df1['Adj. Close'])
	y_train = y[0:train_size]
	y_test = y[train_size:]

	x_test = np.array(df_test)
	del df1['Adj. Close']
	x_train = np.array(df1[0:train_size])

	from sklearn.linear_model import Ridge 
	clf1=Ridge(alpha=1.0)
	print("fitting ridge")
	clf1.fit(x_train, y_train)
	predicted=clf1.predict(x_test)
	print("score")
	confidence1 = clf1.score(x_test, y_test)
	print("Ridge : %.3f%%" % (confidence1*100.0))
	print(" E   N   D")

	clf = LinearRegression()
	print("fitting LR")
	clf.fit(x_train, y_train)
	print("score")
	confidence2 = clf.score(x_test, y_test)
	print("LinearRegressor : %.3f%%" % (confidence2*100.0))
	print(" E   N   D")

	from sklearn.ensemble import BaggingRegressor
	clfy=BaggingRegressor(base_estimator=None,n_estimators=10)
	print("fitting bagging")
	clfy.fit(x_train, y_train)
	predictedy=clfy.predict(x_test)
	print("score")
	confidencey = clfy.score(x_test, y_test)
	print("BAGGING : %.3f%%" % (confidencey*100.0))

	from sklearn.ensemble import GradientBoostingRegressor
	clfz=GradientBoostingRegressor()
	print("GradientBoostingRegressor")
	clfz.fit(x_train, y_train)
	predictedz=clfz.predict(x_test)
	print("score")
	confidencez = clfz.score(x_test, y_test)
	print("BOOSTING : %.3f%%" % (confidencez*100.0))

	import matplotlib.pyplot as plt 
	plt.plot(predictedz,label='predicted')
	plt.plot(y_test,label='Actual')
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.savefig('fea/'+str(company_name)+'.png',dpi=200,bbox_inches='tight')

def timer(sr):
	X = sr.values
	X = X.astype('float32')
	train_size = int(len(X) * 0.80)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	predictions = list()
	months_in_year = 1
	diff = difference(history, months_in_year)
	"""
	autocorrelation_plot(diff)
	plt.show()
	plot_pacf(diff,lags=20)
	plt.show()
	"""
	model = ARIMA(diff, order=(2,1,1))
	
	sr_complete = list()
	for i in range(len(test)):
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		obs = float(test[i])
		history.append(obs)
		sr_complete.append(yhat)
	return sr_complete,train_size
	
def error(train_size,index,x):#call this function to compute error while predicting features 
	set_index=str(index)
	sum=0
	cntr=0
	for i in range(train_size,len(x)):
		if df[set_index][i] != 0:
			vr=(x[i]-df[iset_index][i])/(df[set_index][i])
			sum+=vr
			cntr+=1
	error=(sum/cntr)*100
	print("PercentageError ",error,"%")
	return error

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return dAPL

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


df = timeseries('AAPL') #filename