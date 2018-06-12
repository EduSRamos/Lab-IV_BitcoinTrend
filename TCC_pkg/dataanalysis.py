import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import sklearn.preprocessing as skp
import sklearn.metrics as skm

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms
import arch.unitroot as au
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools

#=================== FUNCTIONS I - Performance Assessment ===================#
def classification_metrics(y_true, y_pred):
    qm = dict()
    qm['TN'], qm['FP'], qm['FN'], qm['TP'] = skm.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    qm['5. Accuracy'] = skm.accuracy_score(y_true=y_true, y_pred=y_pred)
    qm['5. Precision'] = skm.precision_score(y_true=y_true, y_pred=y_pred)
    qm['5. Recall'] = skm.recall_score(y_true=y_true, y_pred=y_pred)
    qm['3. F1'] = skm.f1_score(y_true=y_true, y_pred=y_pred)
    # beta <1/>1 favors precision/recall; beta->0 = only precision; beta->inf = only recall
    qm['2. F-beta_0.5'] = skm.fbeta_score(y_true=y_true, y_pred=y_pred, beta = 0.5)
    # A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction
    # The statistic is also known as the phi coefficient.
    qm['1. Mathews_CorrCoef'] = skm.matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    # k=1: complete agreement / k=0: there is no agreement other than what would be expected by chance
    qm['4. Cohen_Kappa'] = skm.cohen_kappa_score(y1=y_true, y2=y_pred)
    return qm

def acc_weighted(real, pred, decay=0.99):
    '''Calculate the weighted Matthews Correlation Coefficient'''
    weights = np.flip(np.cumprod(np.full(len(real), fill_value=decay)), axis=-1)

    acc_wtd = np.sum((real==pred)*weights) / np.sum(weights)
    return acc_wtd

def mcc_weighted(real, pred, decay=0.99):
    '''Calculate the weighted Matthews Correlation Coefficient'''
    weights = np.flip(np.cumprod(np.full(len(real), fill_value=decay)), axis=-1)

    tp = np.sum( ((real==1) & (pred==1)) *weights) # weighted true positives
    fn = np.sum( ((real==1) & (pred==-1))*weights) # weighted false negatives
    fp = np.sum(((real==-1) & (pred==1)) *weights) # weighted false positives
    tn = np.sum(((real==-1) & (pred==-1))*weights) # weighted true negatives
    assert (tp+tn+fp+fn) - np.sum(weights) < 1e-3

    mcc_wtd = (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return mcc_wtd

#=================== FUNCTIONS II - Time Series Operations ===================#
def difference_ts(df, n=1, columns=None):
    '''Apply differencing to one or more time series.'''
    df = df.dropna(axis=0,how='any')

    if columns is None: columns = list(df.columns)

    df_diff_dict = dict()
    for col in columns:
        df_diff_dict[col] = np.diff(df[col])

    df_diff = pd.DataFrame(data=df_diff_dict, index=df.index[-len(df_diff_dict[col]):])
    return df_diff

def integrate_ts(df, init_val=dict(), n=1, columns=None):
    '''Integrate one or more time series (inverse of differencing).'''
    if columns is None: columns = list(df.columns)

    for i in range(n):
        df_int_dict = dict()
        for col in columns:
            t0 = 0
            if col in init_val and len(init_val[col])>=i:
                t0 = init_val[col][-i]
            df_int_dict[col] = np.cumsum(df[col].values.squeeze()) + t0
        df_int = pd.DataFrame(data=df_int_dict, index=df.index)

    return df_int

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


#================ FUNCTIONS III - Statistical Properties/Tests ================#
def roll_stats(ts,  window=30):
    '''Rolling mean and variance of a time series.'''
    rolling_mean = ts.rolling(window=window, center=False).mean()
    rolling_var = ts.rolling(window=window, center=False).var()

    df_roll = pd.DataFrame(data={'Rolling Mean': rolling_mean, 'Rolling Variance': rolling_var})
    df_roll = df_roll.dropna(axis=0, how='any')
    return df_roll

def test_unitroot_adf(ts, max_lags=90):
    '''Augmented Dickey-Fuller test for presence of unit-root in a time series.'''
    print(au.ADF(ts, max_lags=max_lags, method='AIC'))
    print('\n-----------------------------------------------------\n')

def test_unitroot_phillips_perron(ts):
    '''Phillips-Perron test for presence of unit-root in a time series.'''
    print(au.PhillipsPerron(ts))
    print('\n-----------------------------------------------------\n')

def test_stationarity_kpss(ts):
    '''Kwiatkowski–Phillips–Schmidt–Shin test for stationarity of a time series.'''
    print(au.KPSS(ts))
    print('\n-----------------------------------------------------\n')

def test_serialcorr_breusch_godfrey(df_train, order):
    '''Breusch-Godfrey test for serial correlation of residuals.

       Input:
       - mld: a fitted - statsmodels - model
    '''
    result = sm.tsa.ARMA(df_train, order=order).fit(trend='c',method='css-mle',maxiter=200)
    lm, lmpval, fval, fpval = sms.diagnostic.acorr_breusch_godfrey(results=result, nlags=None, store=False)
    print('\n---------------------  TESTE DE BREUSCH_GODFREY  ---------------------')
    print('HIPOTESE NULA: Nao ha correlacao serial.')
    print('----------------------------------------------------------------------')
    print('Lagrange multiplier test statistic:                 ' + str(lm))
    print('P-value for Lagrange multiplier test:               ' + str(lmpval))
    print('F test statistic (mesmo teste, versao alternativa): ' + str(fval))
    print('P-value for F test:                                 ' + str(fpval))
    print('----------------------------------------------------------------------\n')

def test_serialcorr_ljung_box(ts):
    '''Ljung-Box test for serial correlation of residuals.

       Input:
       - data: a time series
    '''
    lb, lbpval = sms.diagnostic.acorr_ljungbox(ts, lags=None, boxpierce=False)
    print('\n------------------------  TESTE DE LJUNG-BOX  ------------------------')
    print('HIPOTESE NULA: Nao ha correlacao serial.')
    print('----------------------------------------------------------------------')
    print('Min/median/mean/max Ljung-Box test statistic:\n{:.4f} / {:.4f} / {:.4f} / {:.4f}'.format(np.min(lb),
                                                                np.median(lb), np.mean(lb), np.max(lb)))
    print('Min/median/mean/max Ljung-Box p-values:\n{:.4f} / {:.4f} / {:.4f} / {:.4f}'.format(np.min(lbpval),
                                                                np.median(lbpval), np.mean(lbpval), np.max(lbpval)))
    print('Significant (p<0.1) Ljung-Box test statistic:\n{}'.format(lb[lbpval<=0.1]))
    print('Significant (<0.1) p-values for Ljung-Box test:\n{}'.format(lbpval[lbpval<=0.1]))
    print('----------------------------------------------------------------------\n')

def test_cond_het_engle(resid):
    '''Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH effects)'''
    lm, lmpval, fval, fpval = sms.diagnostic.het_arch(resid, maxlag=None, ddof=0)
    print('\n---------  TESTE DE ENGLE PARA HETEROCEDASTICIDADE CONDICIONAL  ----------')
    print('HIPOTESE NULA: Nao ha heterocedasticidade.')
    print('--------------------------------------------------------------------------')
    print('Engle’s Lagrange multiplier test statistic:                 ' + str(lm))
    print('Engle’s p-value for Lagrange multiplier test:               ' + str(lmpval))
    print('Engle’s F test statistic (mesmo teste, versao alternativa): ' + str(fval))
    print('Engle’s p-value for F test:                                 ' + str(fpval))
    print('--------------------------------------------------------------------------\n')

def test_normality_jarque_bera(data):
    '''Jarque-Bera normality test.'''
    jb, p_val, sk, kurt = sms.stattools.jarque_bera(data)
    print('\n-----------------------  TESTE DE JARQUE-BERA  -----------------------')
    print('HIPOTESE NULA:')
    print('\t Os dados foram amostrados de uma distribuicao normal')
    print('----------------------------------------------------------------------')
    print('RESULTADOS:')
    print('\t Estatistica do teste: {:.2f}'.format(jb))
    print('\t P-value do teste: {:.4f}'.format(p_val))
    print('\t Assimetria estimada: {:.4f}'.format(sk))
    print('\t Curtose estimada: {:.4f}'.format(kurt))
    print('----------------------------------------------------------------------\n')

def test_normality_shapiro_wilk(data):
    '''Shapiro-Wilk normality test.'''
    sw, p_val = scs.shapiro(data)
    print('\n-----------------------  TESTE DE SHAPIRO-WILK  ----------------------')
    print('HIPOTESE NULA:')
    print('\t Os dados foram amostrados de uma distribuicao normal')
    print('----------------------------------------------------------------------')
    print('RESULTADOS:')
    print('\t Estatistica do teste: {:.4f}'.format(sw))
    print('\t P-value do teste: {:.4f}'.format(p_val))
    print('----------------------------------------------------------------------\n')


#========================== FUNCTIONS IV - Plotting ===========================#
def plotscatter(df, name='Stock name', title='Stock price', yaxis='Price'):
    '''Plot columns of a DataFrame in the same picture.'''
    columns=df.columns

    # 1) Traces
    df_data = list(go.Scatter(x=df.index, y=df[col], name=col) for col in columns)

    # 2) Layout
    df_layout = go.Layout(title=title, legend={'orientation':'h'}, yaxis={'title':yaxis})

    # 3) Figure
    df_fig = go.Figure(data=df_data, layout=df_layout)
    py.iplot(df_fig)

def candleplot(df, name='Name', cols={}, title='Stock data', yaxis='Price'):
    '''Candle stick plot of financial time series'''
    col_names = {'High':'High', 'Low':'Low', 'Open':'Open', 'Close':'Close'}
    col_names.update(cols)

    trace = go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=name
    )
    layout = go.Layout(
                title=title,
                legend={'orientation':'h'},
                yaxis={'title':yaxis},
                xaxis={'rangeselector':{'buttons':list([{'count':1,
                                                      'label':'1m',
                                                      'step':'month',
                                                      'stepmode':'backward'},
                                                      {'count':6,
                                                      'label':'6m',
                                                      'step':'month',
                                                      'stepmode':'backward'},
                                                      {'count':1,
                                                      'label':'1y',
                                                      'step':'year',
                                                      'stepmode':'backward'},
                                                      {'step':'all'},
                                                     ])
                                         }
                         }
                         )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

# Source: auquan
def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    '''Plot Correlation and Distribution info on Time Series'''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return
