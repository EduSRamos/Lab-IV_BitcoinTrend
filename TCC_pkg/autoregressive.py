import os
import sys

import numpy as np
import pandas as pd

import pickle
import sklearn.preprocessing as skp

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms
import arch.unitroot as au
from arch import arch_model

import operator
import warnings

from . import dataanalysis as da
from . import testspecification as tspec

def execute_test_routine(series_name, df, test_spec, classification_func=None, windows=None, decay_array=None):
    # Fixed parameters
    h = 14 # prediction horizon

    # Parameters
    if windows is None:
        windows = [63, 126, 252, df.shape[0]] # past values to fit ARMA

    # Hyperparameters
    if decay_array is None:
        decay_array = (0.95, 0.98, 0.99, 0.995, 0.999)

    # PART 1: Fit models and make predictions (shortcut for the AR case specifically)
    # Load/predict classes using autoregressive models
    save_pred_path = './Predictions/{}.pkl'.format(series_name)
    try:
        with open(save_pred_path, 'rb') as f:
            df = pickle.load(f)
    except:
        # Prediction function
        df_out = fit_and_predict(df, windows, h)
        df_out.to_pickle(save_pred_path) # backup
        df_out = df_out.dropna()
        df_out = classification_func(df_out)
        df = pd.concat([df, df_out], axis=1)
        df = df.dropna()
        df.to_pickle(save_pred_path)

    # PART #2: Evaluate predictions
    # Model parameters
    ar_params = windows

    # First date of test sets
    start_dates = test_spec.start_dates

    # Initialize variables
    pred_arr = np.empty(0)
    real_arr = np.empty(0)
    for instance in test_spec.instance:
        decay_mcc = {}
        for decay in decay_array:
            decay_mcc[decay] = 0

            # For each forward validation index, get validation-set performance
            for ifv in range(len(instance.expanding_window_fv.train_sets)): # expanding window only
                train_ind = instance.expanding_window_fv.train_sets[ifv]

                # For each combination of parameters, evaluate their weighted-mcc
                params_mcc = {}
                for params in ar_params:
                    real = df.loc[:train_ind[-1], 'Direction'].values
                    pred = df.loc[:train_ind[-1],'{}'.format(params)].values
                    params_mcc[params] = da.mcc_weighted(real, pred, decay)

                # Apply best params to the validation set
                val_ind = instance.expanding_window_fv.val_sets[ifv]
                best_params = max(params_mcc.items(), key=operator.itemgetter(1))[0]
                real = df.loc[val_ind, 'Direction'].values
                pred = df.loc[val_ind,'{}'.format(best_params)].values
                decay_mcc[decay] += da.mcc_weighted(real, pred, decay)

        # Decay that yielded maximum mcc
        best_decay = max(decay_mcc.items(), key=operator.itemgetter(1))[0]
        print('Best decay: {}'.format(best_decay))

        # Use best_decay to select among parameters
        params_mcc = {}
        train_set = instance.train_set
        for params in ar_params:
            real = df.loc[:train_set[-1], 'Direction'].values
            pred = df.loc[:train_set[-1],'{}'.format(params)].values
            params_mcc[params] = da.mcc_weighted(real, pred, best_decay)
        best_params = max(params_mcc.items(), key=operator.itemgetter(1))[0]
        print('Best window: {}'.format(best_params))

        # Predict using parameters
        pred = df.loc[instance.test_set, '{}'.format(best_params)].values.astype(int)
        real = df.loc[instance.test_set,'Direction'].values.astype(int)
        pred_arr = np.r_[pred_arr, pred]
        real_arr = np.r_[real_arr, real]

    # Evaluate the entire test set together
    quality_metrics_dict = da.classification_metrics(y_true=real_arr, y_pred=pred_arr)

    return quality_metrics_dict

def fit_and_predict(df_input, windows, h):

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='once')

    # Input and gets parameters
    ts_in = df_input['Close']
    mean_ts = ts_in.mean()
    std_ts = ts_in.std()

    # Standardizes to mean=0/var=1 and adds 1
    ts_in = (ts_in - mean_ts)/std_ts + 1

    # Create a DataFrame to store results
    windows = np.array(windows)
    mx = max(windows[windows<ts_in.shape[0]]) # largest window excluding the infinite one
    df = pd.DataFrame(columns=['{}'.format(w) for w in windows], index=ts_in.index[mx:])

    # for k in range(0,len(df.index),h):
    for k in range(mx, len(ts_in.index), h): # predict next h values
        print('\n\n',k)
        ts_test  = ts_in.iloc[k:k+h]
        for w in windows:
            print(w, end=': ')
            ts_train = ts_in.iloc[max(k-w,0):k]

            fitted_mdl, best_order, _, _ = best_model_order(ts_train, range(4), range(4))

            mdl_test = sm.tsa.SARIMAX(endog=ts_test, order=best_order, trend='c')
            mdl_test = mdl_test.filter(fitted_mdl.params)

            adj_values = (mdl_test.predict().values - 1)*std_ts + mean_ts
            df.loc[df.index[k-mx:k-mx+h], '{}'.format(w)] = adj_values

    return df

def best_model_order(ts, p_rng, q_rng):
    ts = ts.dropna(axis=0,how='any')
    best_aic = np.inf
    best_order = None
    best_mdl = None

    # Some models raise an exception of dividing by NaN or 0
    np.seterr(divide='ignore', invalid='ignore')

    all_aic = np.full(shape=(max(p_rng)+1,max(q_rng)+1), fill_value=np.nan)
    for i in p_rng:
        for j in q_rng:
            if i is 0 and j is 0: continue
            try:
    #                 tmp_mdl = sm.tsa.ARIMA(ts, order=(i,0,j)).fit(method='mle', trend='c', maxiter=300)
                tmp_mdl = sm.tsa.SARIMAX(ts, order=(i,0,j), trend='c', maxiter=300).fit(solver='lbfgs')
                tmp_aic = tmp_mdl.aic
                all_aic[i,j] = tmp_aic
                if tmp_aic < best_aic:
                    best_mdl = tmp_mdl
                    best_aic = tmp_aic
                    best_order = (i, 0, j)
            except: continue
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    if best_order is None:
        best_order = (1,0,0)
        best_mdl = sm.tsa.SARIMAX(ts, order=best_order, trend='c', maxiter=300).fit(solver='lbfgs')
        print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    return best_mdl, best_order, best_aic, all_aic

def forecast_return_volatility(ts, mu=0, phi=[], theta=[], gmu=0, alpha=[], beta=[]):
    '''1-step ahead prediction for an ARMA-GARCH combination

       Input:
       ts    -- Pandas Series with time series
       mu    -- Mean of ARMA process           (default:  0)
       phi   -- AR coefficients of ARMA model  (default: [])
       theta -- MA coefficients of ARMA model  (default: [])
       gmu   -- Mean of GARCH process          (default:  0)
       alpha -- AR coefficients of GARCH model (default: [])
       beta  -- MA coefficients of GARCH model (default: [])

       Default model is white noise (ARMA(0,0) with mean 0).
       The means, both of ARMA and GARCH, are considered to be 0.

       Output:
       pred -- Pandas DataFrame with 1-step ahead forecasted returns and volatility
    '''
    # Cardinalities
    n = len(ts.values) # number of samples
    p = len(phi)       # number of AR coefficients
    q = len(theta)     # number of MA coefficients
    m = len(alpha)     # number of GARCH-AR coefficients
    s = len(beta)      # number of GARCH-MA coefficients

    # Series (append zeros in beginning for convenience in loop)
    y = ts.values.reshape(n)
    r = np.zeros(shape=n)                     # array with returns
    a = np.zeros(shape=n)                     # array with shocks
    s2 = np.zeros(shape=n)                    # array with sigmas^2
    e = np.full(shape=n, fill_value=np.nan)   # array with eps = standardized shocks = residuals of GARCH

    # Loop over time horizon
    for t in range(n):
        # Predict return in time t
        AR  = np.inner(  phi[0:min(t,p)], np.flip(  y[max(0,t-p):t], axis=-1)).astype(float) # AR term
        MA  = np.inner(theta[0:min(t,q)], np.flip(  a[max(0,t-q):t], axis=-1)).astype(float) # MA term
        r[t] = mu + AR - MA     # add terms to get expected value of ARMA prediction
        a[t] = r[t] - y[t] # residual

        if len(alpha)>0 or len(beta)>0:
            # Evaluate volatility process
            gAR = np.inner(alpha[0:min(t,m)], np.flip(a[max(0,t-m):t]**2, axis=-1)).astype(float) # GARCH-AR term
            gMA = np.inner( beta[0:min(t,s)], np.flip(  s2[max(0,t-s):t], axis=-1)).astype(float) # GARCH-MA term
            s2[t] = gmu + gAR + gMA  # add terms to get volatility

            # Evaluate standardized shocks (residuals of GARCH process)
            assert s2[t]>=0.0
            e[t] = a[t]/np.sqrt(s2[t])

    # Create DataFrame with results
    matrx = np.concatenate((r.reshape(n,1),a.reshape(n,1),(a**2).reshape(n,1),s2.reshape(n,1),e.reshape(n,1)), axis=1)
    pred = pd.DataFrame(data=matrx, index=ts.index,
                        columns=['Pred. Return','Shocks','Shocks^2','Volatility','Std Shocks'])
    return pred
