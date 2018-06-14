""" Technical Analysis Module

Indicators implemented (functions):
    SMA(df_arr, window, col='Close')
    EMA(df_arr, window, col='Close')
    BB(df, col='Close')
    MACD(df, col='Close')
    STOCH(df, cols={})
    RSI(df, col='Close')
    CHAIKIN(df, cols={})
    AROON(df, cols={})
    OBV(df, cols={})
    PVT(df, cols={})
    DISP(df, window, col='Close', ma_func=SMA)
    ROC(df, gap, col='Close')
    WILLIAMS(df, cols={})
    CCI(df, cols={})
"""

import pandas as pd
import numpy as np

import operator

from . import dataanalysis as da
from . import testspecification as tspec

# def execute_test_routine(df, test_spec, ta_params, ta_pred_func):
#     decay_array = (0.95, 0.98, 0.99, 0.995, 0.999)
#     start_dates = test_spec.start_dates
#
#     # Remove NaN entries
#     df = df.dropna()
#
#     # Predict for every combination of parameters
#     for params in ta_params:
#         df['{}'.format(params)] = ta_pred_func(df, *params)
#
#     # Remove NaN entries
#     df = df.dropna()
#
#     # Initialize variables
#     pred_arr = np.empty(0)
#     real_arr = np.empty(0)
#     for instance in test_spec.instance:
#         decay_acc = {}
#         for decay in decay_array:
#             decay_acc[decay] = 0
#
#             # For each forward validation index, get validation-set performance
#             for ifv in range(len(instance.expanding_window_fv.train_sets)): # expanding window only
#                 train_ind = instance.expanding_window_fv.train_sets[ifv]
#
#                 # For each combination of parameters, evaluate their weighted-acc
#                 params_acc = {}
#                 for params in ta_params:
#                     real = df.loc[:train_ind[-1], 'Direction'].values
#                     pred = df.loc[:train_ind[-1],'{}'.format(params)].values
#                     params_acc[params] = da.acc_weighted(real, pred, decay)
#
#                 # Apply best params to the validation set
#                 val_ind = instance.expanding_window_fv.val_sets[ifv]
#                 best_params = max(params_acc.items(), key=operator.itemgetter(1))[0]
#                 real = df.loc[val_ind, 'Direction'].values
#                 pred = df.loc[val_ind,'{}'.format(best_params)].values
#                 decay_acc[decay] += da.acc_weighted(real, pred, decay)
#
#         # Decay that yielded maximum acc
#         best_decay = max(decay_acc.items(), key=operator.itemgetter(1))[0]
#
#         # Use best_decay to select among parameters
#         params_acc = {}
#         train_set = instance.train_set
#         for params in ta_params:
#             real = df.loc[:train_set[-1], 'Direction'].values
#             pred = df.loc[:train_set[-1],'{}'.format(params)].values
#             params_acc[params] = da.acc_weighted(real, pred, best_decay)
#         best_params = max(params_acc.items(), key=operator.itemgetter(1))[0]
#
#         # Predict using parameters
#         pred = df.loc[instance.test_set, '{}'.format(best_params)].values.astype(int)
#         real = df.loc[instance.test_set,'Direction'].values.astype(int)
#         pred_arr = np.r_[pred_arr, pred]
#         real_arr = np.r_[real_arr, real]
#
#     # Evaluate the entire test set together
#     quality_metrics_dict = da.classification_metrics(y_true=real_arr, y_pred=pred_arr)
#
#     return quality_metrics_dict
# def execute_nonparam_test_routine(df, test_spec, ta_pred_func):
#     start_dates = test_spec.start_dates
#
#     # Remove NaN entries
#     df = df.dropna()
#
#     # Predict for only combination of parameters
#     df['Pred'] = ta_pred_func(df)
#
#     # Remove NaN entries
#     df = df.dropna()
#
#     # Initialize variables
#     pred_arr = np.empty(0)
#     real_arr = np.empty(0)
#     for instance in test_spec.instance:
#         pred = df.loc[instance.test_set,'Pred'].values.astype(int)
#         real = df.loc[instance.test_set,'Direction'].values.astype(int)
#         pred_arr = np.r_[pred_arr, pred]
#         real_arr = np.r_[real_arr, real]
#
#     # Evaluate the entire test set together
#     quality_metrics_dict = da.classification_metrics(y_true=real_arr, y_pred=pred_arr)
#
#     return quality_metrics_dict

def SMA_prediction(df, small_window=50, large_window=200):
    """Implements buy/sell strategy based on SMA crossover.

    Keyword arguments:
    df -- DataFrame with column 'Close'
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)
    small_window -- width of small window (default: 50)
    large_window -- width of large window (default: 200)

    Output:
    df_pred -- DataFrame with predictions in column 'SMA_Pred'

    SEE function SMA()
    """

    # Calculate moving averages
    df_sma = SMA(df, small_window, col='Close')
    df_sma['SMA_long'] = SMA(df, large_window, col='Close')
    df_sma = df_sma.dropna()
    df_sma['SMA_diff'] = df_sma['SMA'] -  df_sma['SMA_long']

    df_pred = df_sma[['SMA_diff']].copy()
    df_pred = (df_pred > 0).astype(float) * 2 - 1
    df_pred = df_pred.rename(columns={'SMA_diff':'SMA_Pred'})

    return df_pred
def EMA_prediction(df, small_window=5, large_window=35):
    """Implements buy/sell strategy based on EMA crossover.

    Keyword arguments:
    df -- DataFrame with column 'Close'
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)
    small_window -- width of small window (default: 5)
    large_window -- width of large window (default: 35)

    Output:
    df_pred -- DataFrame with predictions in column 'EMA_Pred'

    SEE function EMA()
    """

    # Calculate moving averages
    df_ema = EMA(df, small_window, col='Close')
    df_ema['EMA_long'] = EMA(df, large_window, col='Close')
    df_ema = df_ema.dropna()
    df_ema['EMA_diff'] = df_ema['EMA'] -  df_ema['EMA_long']

    df_pred = df_ema[['EMA_diff']].copy()
    df_pred = (df_pred > 0).astype(float) * 2 - 1
    df_pred = df_pred.rename(columns={'EMA_diff':'EMA_Pred'})

    return df_pred
def BB_prediction(df, initial_position=1):
    """Implements buy/sell strategy based on the Bollinger Bands.

    Keyword arguments:
    df -- DataFrame with columns {'Close', 'BB_Top', 'BB_Bottom'}
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)

    Output:
    df_pred -- DataFrame with predictions in column 'BB_Pred'

    SEE function BB()
    """

    df_pred = pd.DataFrame(data=np.zeros(df.shape[0]), index=df.index, columns=['BB_Pred'])
    current = initial_position # has bitcoins
    for idx in df.index:
        if df.loc[idx, 'Close'] > df.loc[idx, 'BB_Top']:
            current = -1 # sell bitcoins, the price should fall
        elif df.loc[idx, 'Close'] < df.loc[idx, 'BB_Bottom']:
            current = 1  # buy bitcoins, the price should rise
        df_pred.loc[idx, 'BB_Pred'] = current

    return df_pred
def MACD_prediction(df):
    """Implements buy/sell strategy based on the MACD.

    Keyword arguments:
    df -- DataFrame with column 'MACD_Hist'

    Output:
    df_pred -- DataFrame with predictions in column 'MACD_Pred'

    SEE function MACD()
    """
    df_pred = df[['MACD_Hist']].copy()
    df_pred = (df_pred > 0).astype(float) * 2 - 1
    df_pred = df_pred.rename(columns={'MACD_Hist':'MACD_Pred'})

    return df_pred
def STOCH_prediction(df, buy=20, sell=80, initial_position=1):
    """Implements buy/sell strategy based on the Stochastic Oscillator.

    Keyword arguments:
    df -- DataFrame with column '%D'
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)
    buy -- buy level
    sell -- sell level

    Output:
    stoch_pred -- DataFrame with predictions in column 'STOCH_Pred'

    SEE function STOCH()
    """

    df_pred = pd.DataFrame(data=np.zeros(df.shape[0]), index=df.index, columns=['STOCH_Pred'])
    current = initial_position # has bitcoins?
    for idx in df.index:
        if df.loc[idx, '%D'] > sell:
            current = -1 # sell bitcoins, the price should fall
        elif df.loc[idx, '%D'] < buy:
            current = 1  # buy bitcoins, the price should rise
        df_pred.loc[idx, 'STOCH_Pred'] = current

    return df_pred
def RSI_prediction(df, buy=20, sell=80, initial_position=1):
    """Implements buy/sell strategy based on the RSI.

    Keyword arguments:
    df -- DataFrame with column 'RSI'
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)
    buy -- buy level
    sell -- sell level

    Output:
    df_pred -- DataFrame with predictions in column 'RSI_Pred'

    SEE function RSI()
    """

    df_pred = pd.DataFrame(data=np.zeros(df.shape[0]), index=df.index, columns=['RSI_Pred'])
    current = initial_position # has bitcoins?
    for idx in df.index:
        if df.loc[idx, 'RSI'] > sell:
            current = -1 # sell bitcoins, the price should fall
        elif df.loc[idx, 'RSI'] < buy:
            current = 1  # buy bitcoins, the price should rise
        df_pred.loc[idx, 'RSI_Pred'] = current

    return df_pred
def CHAIKIN_prediction(df):
    """Implements buy/sell strategy based on the CHAIKIN.

    Keyword arguments:
    df -- DataFrame with column 'CHAIKIN'

    Output:
    df_pred -- DataFrame with predictions in column 'MACD_Pred'

    SEE function CHAIKIN()
    """
    if df.loc[df.index[0], 'CHAIKIN'] > 0:
        current = 1   # starts with bitcoins
    else:
        current = -1  # starts without bitcoins

    df_pred = pd.DataFrame(data=np.zeros(df.shape[0]-1), index=df.index[1:], columns=['CHAIKIN_Pred'])
    for i in range(1,df.shape[0]):
        if df.loc[df.index[i-1], 'CHAIKIN'] < 0 and df.loc[df.index[i], 'CHAIKIN'] > 0:
            current = 1 # from < 0 to > 0: buy
        if df.loc[df.index[i-1], 'CHAIKIN'] > 0 and df.loc[df.index[i], 'CHAIKIN'] < 0:
            current = -1 # from > 0 to < 0: sell

        df_pred.loc[df.index[i], 'CHAIKIN_Pred'] = current

    return df_pred
def AROON_prediction(df, buy=70, sell=-70, initial_position=1):
    """Implements buy/sell strategy based on the AROON Oscillator.

    Keyword arguments:
    df -- DataFrame with column 'Aroon_Osc'
    initial_position -- if person begins with BTC (1) or not (-1) (default: 1)
    buy -- buy level
    sell -- sell level

    Output:
    df_pred -- DataFrame with predictions in column 'Aroon_Pred'

    SEE function RSI()
    """

    # Premise: starts with bitcoins
    df_pred = pd.DataFrame(data=np.zeros(df.shape[0]), index=df.index, columns=['Aroon_Pred'])
    current = initial_position # has bitcoins?
    for idx in df.index:
        if df.loc[idx, 'Aroon_Osc'] < sell:
            current = -1 # sell bitcoins, the price should fall
        elif df.loc[idx, 'Aroon_Osc'] > buy:
            current = 1  # buy bitcoins, the price should rise
        df_pred.loc[idx, 'Aroon_Pred'] = current

    return df_pred

def get_indicators(df, indicator_list, cols={}):
    """Calculate a series of indicators.

    Keyword arguments:
    df -- DataFrame with relevant (High, Open, Low, Close, Volume) values
    indicator_list -- list of tuples, in which each tuple has 2 elements:
                      (i) indicators names ('SMA','BB','MACD','RSI',etc) and
                      (ii) keyword parameters (when needed) to use with indicat.
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'High'/'Low'/'Close'/'Volume'/'Open' as keys

    Output:
    df_out -- DataFrame with columns corresponding to the indicators
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('get_indicators(): df must be a pandas DataFrame')

    col_names={'High':'High','Low':'Low','Open':'Open','Close':'Close','Volume':'Volume'}
    col_names.update(cols)

    func_lookup = {'SMA':SMA, 'EMA':EMA, 'BB':BB, 'MACD':MACD, 'STOCH':STOCH,
                   'RSI':RSI, 'CHAIKIN':CHAIKIN, 'AROON':AROON, 'OBV':OBV,
                   'PVT':PVT, 'DISP':DISP, 'ROC':ROC, 'WILLIAMS':WILLIAMS,
                   'CCI':CCI
    }

    df_out = pd.DataFrame(data=[], columns=[], index=df.index)
    for (indicator, kwparams) in indicator_list:
        # print(indicator, kwparams)
        df_out = pd.concat([df_out, func_lookup[indicator](df, **kwparams)], axis=1)

    return df_out


# Simple Moving Average - Dependencies: last <window-1> values / Range: inf
def SMA(df_arr, window, col='Close', output_colname='SMA'):
    """Calculate the Simple Moving Average.

    Keyword arguments:
    df_arr -- DataFrame or numpy array
    window -- Window of past values to use
    col -- Column with values (default: Close)
    output_colname -- Name of column that will be returned

    Output:
    sma -- DataFrame with column output_colname or a numpy array
    """
    if isinstance(df_arr, pd.DataFrame):
        values = df_arr[col]
    elif isinstance(df_arr, np.ndarray):
        values = df_arr
    else:
        raise TypeError('df_arr must be a DataFrame or a numpy array')

    weights = np.repeat(1.0, window)/window

    if isinstance(df_arr, pd.DataFrame):
        sma = pd.DataFrame(data=np.convolve(values, weights, 'valid'),
                           index=df_arr.index[window-1:],
                           columns=[output_colname]
                          )
    else:
        sma = np.convolve(values, weights, 'valid')
    return sma
# Exponential Moving Average - Dependencies: last <window-1> values / Range: inf
def EMA(df_arr, window, col='Close', output_colname='EMA'):
    """Calculate the Exponential Moving Average.

    Keyword arguments:
    df_arr -- pandas DataFrame or numpy array
    window -- Window of past values to use
    col -- Column with values (default: Close)
    output_colname -- Name of column that will be returned

    Output:
    ema -- DataFrame with column output_colname or a numpy array
    """
    if isinstance(df_arr, pd.DataFrame):
        values = df_arr[col]
    elif isinstance(df_arr, np.ndarray):
        values = df_arr
    else:
        raise TypeError('df_arr must be a DataFrame or a numpy array')

    weight = 2.0/(1+window)
    ema_arr = np.zeros(len(values)-window+1)
    ema_arr[0] = np.mean(values[:window]) # SMA seed
    for i in np.arange(1, len(ema_arr)):
        ema_arr[i] = weight*values[i+window-1] + (1-weight)*ema_arr[i-1]

    if isinstance(df_arr, pd.DataFrame):
        ema = pd.DataFrame(data=ema_arr, index=df_arr.index[window-1:], columns=[output_colname])
    else:
        ema = ema_arr
    return ema
# Bollinger Bands: Dependencies: last 19 values / Range: unbounded
def BB(df, col='Close'):
    """Calculate the Bollinger Bands.

    Keyword arguments:
    df -- DataFrame
    col -- Column with values (default: Close)

    Output:
    bb -- DataFrame with bands in columns {'BB_Top', 'BB_Middle', 'BB_Bottom'}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('BB: df must be a pandas DataFrame')

    middle_band = SMA(df, 20, col=col)['SMA']
    sd_20 = np.zeros(len(middle_band))
    for i in np.arange(20,len(df)+1):
        sd_20[i-20] = np.std(df[col].values[i-20:i])

    bb_dict = {'BB_Middle': middle_band,
               'BB_Top': (middle_band + 2*sd_20),
               'BB_Bottom': (middle_band - 2*sd_20)
              }
    bb = pd.DataFrame(data=bb_dict, index=df.index[19:])
    return bb
# MA Convergence Divergence - Dependencies: last 32 values / Range: unbounded
def MACD(df, col='Close'):
    """Calculate the MACD - Moving Average Convergence Divergence.

    Keyword arguments:
    df -- DataFrame
    col -- Column with values (default: Close)

    Output:
    macd -- DataFrame with columns {'MACD_Line', 'MACD_SigLine', 'MACD_Hist'}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('MACD: df must be a pandas DataFrame')

    values = np.array(df[col].values)
    ema_26 = EMA(values, 26)
    ema_12 = EMA(values, 12)

    macd_line = ema_12[14:] - ema_26

    # Include NaNs to fill array
    signal_line = np.r_[np.full(8,fill_value=np.nan), EMA(macd_line, 9)]
    macd_hist = macd_line - signal_line

    macd_dict = {'MACD_Line':macd_line, 'MACD_SigLine':signal_line, 'MACD_Hist':macd_hist}
    macd = pd.DataFrame(data=macd_dict, index=df.index[-len(macd_line):])
    return macd
# Stochastic Oscillator- Dependencies:last 13/16 values (%K/%D) / Range:[0, 100]
def STOCH(df, cols={}):
    """Calculate the Stochastic Oscillator.

    Keyword arguments:
    df -- DataFrame with High, Low and Close values
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'High'/'Low'/'Close' as keys

    Output:
    stoch -- DataFrame with columns {'%K', '%D'}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('STOCH: df must be a pandas DataFrame')

    col_names={'High':'High','Low':'Low','Close':'Close'}
    col_names.update(cols)

    # Lowest Low and Highest High: last 14 dias (including today)
    # %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    # %D = 3-day SMA of %K
    highest_arr = np.zeros(len(df)-13)
    lowest_arr = np.zeros(len(df)-13)
    for i in np.arange(13,len(df)):
        highest_arr[i-13] = np.max(df[col_names['High']][i-13:i+1])
        lowest_arr[i-13] = np.min(df[col_names['Low']][i-13:i+1])
    highest = pd.Series(data=highest_arr, index=df.index[13:])
    lowest = pd.Series(data=lowest_arr, index=df.index[13:])

    K_ser = (df[col_names['Close']][13:] - lowest)/(highest-lowest) * 100
    D_arr = np.r_[np.full(2,np.nan), EMA(K_ser.values, 3)]

    stoch_dict = {'%K': K_ser, '%D':D_arr}
    stoch = pd.DataFrame(data=stoch_dict, index=K_ser.index)
    return stoch
# Relative Strength Index - Dependencies: last 14 values / Range: [0, 100]
def RSI(df, col='Close'):
    """Calculate the Relative Strength Index.

    Keyword arguments:
    df -- DataFrame
    col -- Column with values (default: Close)

    Output:
    rsi -- DataFrame with column 'RSI'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('RSI: df must be a pandas DataFrame')

    values = df[col].values

    # Calcula ganhos/perdas
    delta_values = values[1:] - values[:-1]

    # Vetor
    n_rsi = len(values)-14
    rsi_arr = np.zeros(n_rsi)

    # Faz primeiro calculo (medias dos primeiros 14 periodos), avalia RS(0) e RSI(0)
    avg_gain = np.mean(np.maximum(delta_values[:14],0))
    avg_loss = np.abs(np.mean(np.minimum(delta_values[:14],0)))
    rsi_arr[0] = 100 - 100/(1 + avg_gain/max(avg_loss,1e-5))

    # Itera para calcular ganho_medio(t), perda_media(t), RS(t), RSI(t)
    for i in np.arange(1, n_rsi):
        avg_gain = (avg_gain*13 + max(delta_values[13+i],0)) / 14
        avg_loss = (avg_loss*13 + np.abs(min(delta_values[13+i],0))) / 14
        rsi_arr[i] = 100 - 100/(1 + avg_gain/max(avg_loss,1e-5))

    rsi = pd.DataFrame(data=rsi_arr, index=df.index[14:], columns=['RSI'])
    return rsi
# Chaikin Oscillator - Dependencies: last 10 values / Range: REALLY unbounded
def CHAIKIN(df, cols={}):
    """Calculate the Accumulation Distribution Line and Chaikin Oscillator.

    Keyword arguments:
    df -- DataFrame with High, Low, Close and Volume values
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'High'/'Low'/'Close'/'Volume' as keys

    Output:
    chaikin -- DataFrame with column 'CHAIKIN'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('CHAIKIN: df must be a pandas DataFrame')

    col_names={'Close':'Close','Low':'Low','High':'High','Volume':'Volume'}
    col_names.update(cols)

    # Money flow multiplier
    close,low,high,volume = df[col_names['Close']],df[col_names['Low']],df[col_names['High']],df[col_names['Volume']]
    mfm = ((close-low) - (high-close))/(high-low)

    # Money flow volume
    mfv = mfm * volume

    # A/D Line
    adl_arr = (mfv.cumsum()).values

    # Chaikin Oscillator
    chaikin_arr = EMA(adl_arr, 3)[7:] - EMA(adl_arr,10)
    chaikin = pd.DataFrame(data=chaikin_arr, index=df.index[9:], columns=['CHAIKIN'])

    return chaikin
# Aroon Oscillator - Dependencies: last 23 values / Range: [-100 100]
def AROON(df, cols={}):
    """Calculate the Accumulation Distribution Line.

    Keyword arguments:
    df -- DataFrame
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'High'/'Low' as keys

    Output:
    stoch -- DataFrame with columns {'Aroon_Up', 'Aroon_Down', 'Aroon-Osc'}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('AROON: df must be a pandas DataFrame')

    col_names={'Low':'Low','High':'High'}
    col_names.update(cols)

    # Aroon Up = 100 x (25 - Days Since 25-day High)/25
    # Aroon Down = 100 x (25 - Days Since 25-day Low)/25
    # Aroon Oscillator = Aroon-Up  -  Aroon-Down
    t = 24
    aroon_up = np.zeros(len(df)-t+1)
    aroon_down = np.zeros(len(df)-t+1)
    for i in np.arange(t,len(df)+1):
        aroon_up[i-t] = 100 * (float(df[col_names['High']].values[i-t:i].argmax())+1) / t
        aroon_down[i-t] = 100 * (float(df[col_names['Low']].values[i-t:i].argmin())+1) / t
    aroon_dict = {'Aroon_Up': aroon_up, 'Aroon_Down': aroon_down, 'Aroon_Osc':(aroon_up-aroon_down)}
    aroon = pd.DataFrame(data=aroon_dict, index=df.index[t-1:])
    return aroon
# On-Balance Volume - Dependencies: last 1 day / Range: unbounded
def OBV(df, cols={}):
    """Calculate the On-Balance Volume.

    Keyword arguments:
    df -- DataFrame
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'Close'/'Volume' as keys

    Output:
    stoch -- DataFrame with column 'OBV'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('OBV: df must be a pandas DataFrame')

    col_names = {'Close':'Close', 'Volume':'Volume'}
    delta = np.r_[0, df[col_names['Close']].values[1:] - df[col_names['Close']].values[:-1]]
    delta[delta<0] = -1
    delta[delta>0] = 1
    delta[delta==0] = 0
    obv = np.multiply(df[col_names['Volume']].values, delta)
    obv = obv.cumsum()

    obv = pd.DataFrame(data=obv, index=df.index, columns=['OBV'])
    return obv
# Price-Volume Trend - Dependencies: last 1 value / Range: unbounded
def PVT(df, cols={}):
    col_names = {'Close':'Close', 'Volume':'Volume'}
    col_names.update(cols)

    closes = df[col_names['Close']].values
    volumes = df[col_names['Volume']].values

    pvt_arr = np.r_[0, ((closes[1:]-closes[:-1])/closes[:-1])*volumes[1:]]
    pvt_arr = np.cumsum(pvt_arr)

    pvt = pd.DataFrame(data=pvt_arr, index=df.index, columns=['PVT'])

    return pvt
# Disparity - Dependencies: last <window-1> values / Range: [0, inf] (but ~ 1)
def DISP(df, window, col='Close', ma_func=SMA, output_colname='DISP'):
    """Calculate the disparity indicator

    Keyword arguments:
    df -- DataFrame
    window -- Window of past values to use
    col -- Column with values (default: Close)
    ma_func -- A function that calculates a moving average (default: SMA)
    output_colname -- Name of column that will be returned

    Output:
    disp -- DataFrame with column <output_colname>
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('DISP: df must be a pandas DataFrame')

    sma_arr = ma_func(df,window,col).values.ravel()
    disp_arr = df[col].values[window-1:] / sma_arr

    disp = pd.DataFrame(data=disp_arr, index=df.index[window-1:], columns=[output_colname])

    return disp
# Rate of Change - Dependencies: last <window> values / Range: [0, inf] (but ~1)
def ROC(df, gap, col='Close', output_colname='ROC'):
    """Calculate the rate of change

    Keyword arguments:
    df -- DataFrame
    gap -- Distance between values to compare
    col -- Column with values (default: Close)
    output_colname -- Name of column that will be returned

    Output:
    roc -- DataFrame with column output_colname
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('ROC: df must be a pandas DataFrame')

    values = df[col].values
    roc_arr = values[gap:] / values[:-gap]

    roc = pd.DataFrame(data=roc_arr, index=df.index[gap:], columns=[output_colname])

    return roc
# Larry William's %R - Dependencies: past 13 days / Range: [0 -100]
def WILLIAMS(df, cols={}):
    """Calculate the disparity indicator

    Keyword arguments:
    df -- DataFrame
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'Close'/'High'/'Low' as keys

    Output:
    williams -- DataFrame with column 'WILLIAMS_%R'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('WILLIAMS: df must be a pandas DataFrame')

    gap = 14 # default value
    col_names = {'High':'High', 'Low':'Low', 'Close':'Close'}
    col_names.update(cols)

    closes = df[col_names['Close']].values
    highs = df[col_names['High']].values
    lows = df[col_names['Low']].values

    williams_arr = np.zeros(len(closes)-gap+1)
    for i in range(len(williams_arr)):
        highest = np.max(highs[i:i+gap])
        lowest = np.min(lows[i:i+gap])
        williams_arr[i] = -100*(highest - closes[i+gap-1]) / (highest-lowest)

    williams = pd.DataFrame(data=williams_arr, index=df.index[13:], columns=['Williams_%R'])

    return williams
# Commodity Channel Index - Dependencies: last 19 values / Range: unbounded
def CCI(df, cols={}):
    """Calculate the commodity channel index

    Keyword arguments:
    df -- DataFrame
    cols -- Dictionary with column names that are not default (default:empty)
            Use the default names 'Close'/'High'/'Low' as keys

    Output:
    cci -- DataFrame with column 'CCI'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('CCI: df must be a pandas DataFrame')

    col_names = {'High':'High', 'Low':'Low', 'Close':'Close'}

    closes = df[col_names['Close']].values
    highs = df[col_names['High']].values
    lows = df[col_names['Low']].values

    # 1) Typical Prices
    tp = (closes+highs+lows) / 3

    # 2) 20-day Moving Average of tp
    tp_sma20 = SMA(tp, 20)

    # 3) Constant
    cte = 0.015

    # 4) Mean deviation
    mean_dev = np.zeros(tp_sma20.shape)
    for i in range(len(tp_sma20)):
        mean_dev[i] = np.mean(np.abs(tp[i:i+20] - tp_sma20[i]))

    # Commodity Channel Index
    cci_arr = (tp[19:] - tp_sma20) / (cte * mean_dev)
    cci = pd.DataFrame(data=cci_arr, index=df.index[19:], columns=['CCI'])

    return cci


class TechAnalysis:
    '''Technical Analysis class for '''
    calc_funcs_dict =  {'STOCH': STOCH,
                        'RSI': RSI,
                        'AROON': AROON,
                        'BB': BB,
                        'MACD': MACD,
                        'CHAIKIN': CHAIKIN }

    param_funcs_dict = {'SMA': SMA_prediction,
                        'EMA': EMA_prediction,
                        'STOCH': STOCH_prediction,
                        'RSI': RSI_prediction,
                        'AROON': AROON_prediction }

    nonparam_funcs_dict = {'BB': BB_prediction,
                           'MACD': MACD_prediction,
                           'CHAIKIN': CHAIKIN_prediction }

    def __init__(self, df, test_spec):
        self.df = df.copy()
        self.test_spec = test_spec
        self.params_dict = {'SMA': [(50,200), (30,70), (10,30)],
                            'EMA': [(10,50),(5,35),(3,20)],
                            'STOCH': [(20,80), (30,70), (40,60)],
                            'RSI': [(20,80), (30,70), (40,60)],
                            'AROON': [(90,-90), (70,-70), (30,-30)]
                            }

    def get_df(self, key):
        df = self.df.copy()
        if key in self.calc_funcs_dict:
            df= pd.concat([df, self.calc_funcs_dict[key](df)], axis=1)
        return df
    def execute_test_routine(self):
        ta_res = {}

        for key, func in self.param_funcs_dict.items():
            df = self.get_df(key)
            ta_res[key] = self.execute_param_test_routine(df,
                          ta_params=self.params_dict[key], ta_pred_func=func)

        for key, func in self.nonparam_funcs_dict.items():
            df = self.get_df(key)
            ta_res[key] = self.execute_nonparam_test_routine(df, ta_pred_func=func)

        self.results = pd.DataFrame(data=ta_res)
    def execute_param_test_routine(self, df, ta_params, ta_pred_func):
        test_spec = self.test_spec
        decay_array = (0.95, 0.98, 0.99, 0.995, 0.999)
        start_dates = test_spec.start_dates

        # Remove NaN entries
        df = df.dropna()

        # Predict for every combination of parameters
        for params in ta_params:
            df['{}'.format(params)] = ta_pred_func(df, *params)

        # Remove NaN entries
        df = df.dropna()

        # Initialize variables
        pred_arr = np.empty(0)
        real_arr = np.empty(0)
        for instance in test_spec.instance:
            decay_acc = {}
            for decay in decay_array:
                decay_acc[decay] = 0

                # For each forward validation index, get validation-set performance
                for ifv in range(len(instance.expanding_window_fv.train_sets)): # expanding window only
                    train_ind = instance.expanding_window_fv.train_sets[ifv]

                    # For each combination of parameters, evaluate their weighted-acc
                    params_acc = {}
                    for params in ta_params:
                        real = df.loc[:train_ind[-1], 'Direction'].values
                        pred = df.loc[:train_ind[-1],'{}'.format(params)].values
                        params_acc[params] = da.acc_weighted(real, pred, decay)

                    # Apply best params to the validation set
                    val_ind = instance.expanding_window_fv.val_sets[ifv]
                    best_params = max(params_acc.items(), key=operator.itemgetter(1))[0]
                    real = df.loc[val_ind, 'Direction'].values
                    pred = df.loc[val_ind,'{}'.format(best_params)].values
                    decay_acc[decay] += da.acc_weighted(real, pred, decay)

            # Decay that yielded maximum acc
            best_decay = max(decay_acc.items(), key=operator.itemgetter(1))[0]

            # Use best_decay to select among parameters
            params_acc = {}
            train_set = instance.train_set
            for params in ta_params:
                real = df.loc[:train_set[-1], 'Direction'].values
                pred = df.loc[:train_set[-1],'{}'.format(params)].values
                params_acc[params] = da.acc_weighted(real, pred, best_decay)
            best_params = max(params_acc.items(), key=operator.itemgetter(1))[0]

            # Predict using parameters
            pred = df.loc[instance.test_set, '{}'.format(best_params)].values.astype(int)
            real = df.loc[instance.test_set,'Direction'].values.astype(int)
            pred_arr = np.r_[pred_arr, pred]
            real_arr = np.r_[real_arr, real]

        # Evaluate the entire test set together
        quality_metrics_dict = da.classification_metrics(y_true=real_arr, y_pred=pred_arr)

        return quality_metrics_dict
    def execute_nonparam_test_routine(self, df, ta_pred_func):
        test_spec = self.test_spec
        start_dates = test_spec.start_dates

        # Remove NaN entries
        df = df.dropna()

        # Predict for only combination of parameters
        df['Pred'] = ta_pred_func(df)

        # Remove NaN entries
        df = df.dropna()

        # Initialize variables
        pred_arr = np.empty(0)
        real_arr = np.empty(0)
        for instance in test_spec.instance:
            pred = df.loc[instance.test_set,'Pred'].values.astype(int)
            real = df.loc[instance.test_set,'Direction'].values.astype(int)
            pred_arr = np.r_[pred_arr, pred]
            real_arr = np.r_[real_arr, real]

        # Evaluate the entire test set together
        quality_metrics_dict = da.classification_metrics(y_true=real_arr, y_pred=pred_arr)

        return quality_metrics_dict
