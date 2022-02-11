from scipy.optimize import leastsq, curve_fit, minimize, OptimizeResult
import matplotlib
from matplotlib import axes
import matplotlib.pyplot as plt 
import numpy as np
import math
from typing import Callable
import datetime
import pandas as pd 
from io import StringIO
from numpy import mean, std, median


def f_logistic(x:np.ndarray, A, B, C, D) -> np.ndarray:
    return (A - D)/(1 + (x/C)**B) + D

def loss_logistic(p, y, x):
    A, B, C, D = p
    return np.sum((y - f_logistic(x, A, B, C, D))**2)

def f_gompertz(x:np.ndarray, A, B, C, D) -> np.ndarray:
    # return D + A * (np.exp(np.exp(B * (C * x))))
    return D + C * np.exp(-B * np.exp(-x / A))

def loss_gompertz(p, y, x):
    A, B, C, D = p
    return np.sum((y - f_gompertz(x, A, B, C, D))**2)

def fit(x:np.ndarray, y:np.ndarray, lossFunc:Callable) -> OptimizeResult:
    """Tries to fit x and y data to using given loss function.
       loss function itself contains the function to be fit

    Args:
        x (np.ndarray): x data
        y (np.ndarray): y data
        lossFunc (function): loss function

    Returns:
        OptimizeResult: scipy OptimizeResult object. Member x is numpy.ndarray 
                        which contains the optimization solution.
    """
    A0 = y.max() * 2
    D0 = y.min() / 2
    C0 = x.mean() * 2
    B0 = 1
    p0 = [A0, B0, C0, D0] # starting values to begin optimization.
    r = minimize(lossFunc, x0=p0, args=(y, x), method='CG')
    return r

def plotfunc(xrange:tuple, f:Callable, r:OptimizeResult, axs:matplotlib.axes) -> tuple:
    xp = np.linspace(xrange[0], xrange[1], 100)
    yp = f(xp, *r.x)
    axs.plot(xp, yp)
    return xp, yp

def plotdata(x:np.ndarray, y:np.ndarray, axs:matplotlib.axes, xrange=(-1,-1)) -> tuple:
    xmin = xrange[0]
    xmax = xrange[1]
    if xmin == -1:
        xmin = x.min()
    if xmax == -1:
        xmax = x.max()
    x, y = x[x >= xmin], y[x >= xmin]
    x, y = x[x <= xmax], y[x <= xmax]
    axs.scatter(x, y)
    return np.array(x), np.array(y)


def doublingt(xrange:tuple, f:Callable, r:OptimizeResult, axs:matplotlib.axes) -> tuple:
    """Plots doubling time chart in semi-log scale (log y - linear x axis).
       Returns x and y lists in a tuple.
       Time point for minimum doubling time can be retrieved by:
       dx[dy.argmin()]

    Args:
        xrange (tuple): (xmin, xmax)
        f (Callable): fit function
        r (OptimizeResult): optimization results
        axs (matplotlib.axes): the axis for the plot

    Returns:
        tuple: x and y as lists
    """
    xp = np.linspace(xrange[0], xrange[1], 100)
    yp = f(xp, *r.x)
    dy = []
    for i in range(0, len(xp)-1):
        dx = xp[i+1] - xp[i]
        _dy = math.log(2) * dx / (math.log(yp[i+1]) - math.log(yp[i])) 
        dy.append(_dy)
    axs.set_yscale('log')
    axs.minorticks_on()
    axs.yaxis.set_minor_locator(plt.MaxNLocator(4))
    axs.grid(b=True, which='major', color='#666666', linestyle='-')
    axs.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs.plot(xp[:-1], dy, c='g')
    return np.array(xp[:-1]), np.array(dy)
    

def timestr(s:str, century='20') -> datetime:
    """Converts YYMMDD-hhmm to datetime.datetime object. 
       century is set to 20 as default

    Args:
        s (str): date signature string YYMMDDhhmm
        century (str): 2-digit century string. 20 as default

    Returns:
        datetime: datetime.datetime object
    """
    return datetime.datetime(
        int('{0}{1}'.format(century, s[0:2])),  # year
        int(s[2:4]),                            # month
        int(s[4:6]),                            # day
        int(s[7:9]),                            # hr
        int(s[9:11])                            # min
    )

def dt(t0:str, t1:str) -> int:
    """Delta t as minutes between t0 and t1 date-time strings

    Args:
        t0 (str): date-time string in YYMMDD-hhmm format
        t1 (str): date-time string in YYMMDD-hhmm format

    Returns:
        int: delta t in minutes
    """
    return (timestr(t1).timestamp() - timestr(t0).timestamp()) / 60

def timeSeries(x:np.ndarray, y:np.ndarray):
    idx = (~np.isnan(y)) # eliminate missing data points
    x = x[idx]
    y = y[idx] 
    floatX = [0]
    t0 = x[0]
    for i in range(1, len(x)):
        floatX.append(floatX[i-1] + dt(t0, x[i]))
        t0 = x[i]
    x = np.array(floatX)
    
    return x, y


def readPlates(fn:str):
    f = open(fn, 'r')
    layout = []
    plates = []
    plateNames = []
    tempFile = StringIO('')
    line = f.readline()
    header = ''
    while line:
        if '#' in line:
            if header != '':
                tempFile.flush()
                tempFile.seek(0)
                #print(header)
                if header == 'layout':
                    df = pd.read_csv(tempFile, sep='\t', header=None)
                    layout = df.values.flatten().tolist()
                else:
                    df = pd.read_csv(tempFile, sep='\t', header=None)
                    plates.append(df.values.flatten().tolist())
                    plateNames.append(header)
                tempFile = StringIO('')
            header = line[1:].strip()
        else:
            tempFile.write(line)
        line = f.readline()
    if header != '':
        tempFile.flush()
        tempFile.seek(0)
        #print(header)
        if header == 'layout':
            df = pd.read_csv(tempFile, sep='\t', header=None)
            layout = df.values.flatten().tolist()
        else:
            df = pd.read_csv(tempFile, sep='\t', header=None)
            plates.append(df.values.flatten().tolist())
            plateNames.append(header)
    # print(layout)
    # for pn, p in zip(plateNames, plates):
    #     print(pn)
    #     print(p)
    #     print('\n')
    df = None
    groups = {}
    for w in range(0, len(layout)):
        if w != '':  # is this well occupied
            dflist = []
            t0 = ''
            for pn, p in zip(plateNames, plates):
                if t0 == '':
                    t0 = pn
                sampleName = '{0}.{1}'.format(layout[w], str(w))
                dflist.append({'t': dt(t0, pn), sampleName: p[w]})
                if not layout[w] in groups:
                    groups[layout[w]] = [sampleName]
                else:
                    if not sampleName in groups[layout[w]]:
                        groups[layout[w]].append(sampleName)
            if df is None:
                df = pd.DataFrame(dflist)
            else:
                df = pd.merge(left=df, right=pd.DataFrame(dflist), left_on='t', right_on='t')
    return df, groups


def growthAnalysis(plate:pd.DataFrame, groups:dict, outdir:str) -> pd.DataFrame:
    growthParams = []
    for k in groups.keys():
        print(k)
        fig = plt.figure()
        fig.set_size_inches(w=16, h=8)
        axsGrowth = fig.add_subplot(121)
        axsDoubling = fig.add_subplot(122)
        growthPlateauSeries = []
        minDoubTYSeries = []
        minDoubTXSeries = []
        for v in groups[k]:
            x = plate['t']
            y = plate[v]
            r = fit(x, y, loss_logistic)
            plotdata(x, y, axs=axsGrowth, xrange=(0, 1500))
            fx, fy = plotfunc((0, 1500), f_logistic, r, axs=axsGrowth)
            dx, dy = doublingt((0, 1500), f_logistic, r, axs=axsDoubling)
            growthPlateauSeries.append(fy.max()) # calculated max OD600 value 
            minDoubTYSeries.append(dy[dy.argmin()]) 
            minDoubTXSeries.append(dx[dy.argmin()])
        fig.savefig('{0}/{1}.png'.format(outdir, k))
        plt.close(fig)
        growthParams.append(
            {
                'sample': k,
                'growth plateau min': min(growthPlateauSeries),
                'growth plateau max': max(growthPlateauSeries),
                'mean growth plateau': mean(growthPlateauSeries),
                'std growth plateau': std(growthPlateauSeries),
                'min peak growth rate t': min(minDoubTXSeries),
                'max peak growth rate t': max(minDoubTXSeries),
                'mean peak growth rate t': mean(minDoubTXSeries),
                'std peak growth rate t': std(minDoubTXSeries),
                'min shortest doubling time': min(minDoubTYSeries),
                'max shortest doubling time': max(minDoubTYSeries),
                'mean shortest doubling time': mean(minDoubTYSeries),
                'std shortest doubling time': std(minDoubTYSeries)
            }
        )
    return pd.DataFrame(growthParams)
