import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
from scipy.stats import binned_statistic
from ipywidgets import widgets
from IPython.display import display
import time
import matplotlib.ticker as ticker


# Authors: Andjelka Kovacevic, Isidora Jankov & Viktor Radovic


def LC_conti(T, deltatc=1, oscillations=True, A=0.14, noise=0.00005, z=0, frame='observed'):
    """ 
    Generate one artificial light curve using a stochastic model based on the Damped random walk (DRW)
    proccess. Parameters describing the model are characteristic amplitude ("logsig2" in code) and time 
    scale of the exponentially-decaying variability ("tau" in code), both infered from physical 
    quantities such are supermassive black hole mass and/or luminosity of the AGN. For further details
    regarding the model see Kovačević et al. (2021) and references therein.
    
    Parameters:
    -----------
    T: int
        Total time span of the light curve. It is recommended to generate light curves to be at least 
        10 times longer than their characteristic timescale (Kozłowski 2017). 
    deltatc: int, default=1
        Cadence (or sampling rate) - time interval between two consecutive samplings of the light 
        curve in days.
    oscillations: bool, default=True
        If True, light curve simulation will take an oscillatory signal into account.
    A: float, default=0.14
        Amplitude of the oscillatory signal in magnitudes (used only if oscillations=True).
    noise: float, default=0.00005
        Amount of noise to include in the light curve simulation.
    z: float, default=0
        Redshift.
    frame: {'observed', 'rest'}, default='observed'
        Frame of reference.
    
    Returns:
    --------
    tt: np.array
        Days when the light curve was sampled.
    yy: np.array
        Magnitudes of the simulated light curve.
        
    References:
    -----------
    Ivezić, Ž., et al. 2019, ApJ, 873, 111 (https://iopscience.iop.org/article/10.3847/1538-4357/ab042c)
    Kelly, B.C., Bechtold, J., & Siemiginowska, A. 2009, ApJ, 698, 895 (https://iopscience.iop.org/article/10.1088/0004-637X/698/1/895)
    Kovačević, A., et al. 2021, submitted to MNRAS (https://github.com/LSST-sersag/white_paper/blob/main/data/paper.pdf)
    Kozłowski, S. 2017, A&A, 597, A128 (https://www.aanda.org/articles/aa/full_html/2017/01/aa29890-16/aa29890-16.html)
    """
    
    # Constants
    const1 = 0.455*1.25*1e38
    const2 = np.sqrt(1e09)
    meanmag = 23.
    
    # Generating survey days 
    tt = np.arange(0, T, int(deltatc))
    times = tt.shape[0]
    
    # Generating log L_bol
    loglumbol = np.random.uniform(42.2,49,1)
    lumbol = np.power(10,loglumbol)

    # Calculate M_{SMBH}
    msmbh=np.power((lumbol*const2/const1),2/3.)
    
    # Calculate damping time scale (Eq 22, Kelly et al. 2009)
    logtau = -8.13+0.24*np.log10(lumbol)+0.34*np.log10(1+z)
    if frame == 'observed':
        # Convering to observed frame (Eq 17, Kelly et al. 2009)
        tau = np.power(10,logtau)*(1+z)
    elif frame == 'rest':
        tau = np.power(10,logtau)
    
    # Calculate log sigma^2 - an amplitude of correlation decay (Eq 25, Kelly et al. 2009)
    logsig2 = 8-0.27*np.log10(lumbol)+0.47*np.log10(1+z)
    if frame == 'observed':
        # Convering to observed frame (Eq 18, Kelly et al. 2009)
        sig = np.sqrt(np.power(10,logsig2))/np.sqrt(1+z)
    elif frame == 'rest':
        sig = np.sqrt(np.power(10,logsig2))
          
    # OPTIONAL: Calculate the broad line region radius
    logrblr=1.527+0.533*np.log10(lumbol/1e44)
    rblr=np.power(10,logrblr)
    rblr=rblr/10
    
    # Calculating light curve magnitudes
    ss = np.zeros(times)
    ss[0] = meanmag # light curve is initialized
    SFCONST2=sig*sig
    ratio = -deltatc/tau

    for i in range(1, times):
        ss[i] = np.random.normal(ss[i-1]*np.exp(ratio) + meanmag*(1-np.exp(ratio)),
                                     np.sqrt(10*0.5*tau*SFCONST2*((1-np.exp(2*ratio)))),1)
        
    # Calculating error (Ivezic et al. 2019)
    gamma=0.039
    m5=24.7
    x=np.zeros(ss.shape)
    x=np.power(10, 0.4*(ss-m5))

    err = (0.005*0.005) + (0.04-gamma)*x + gamma*x*x
    
    # Final light curve with oscillations
    if oscillations == True:
        # Calculate underlying periodicity
        conver=173.145 # convert from LightDays to AU
        lightdays=10
        P = np.sqrt(((lightdays*conver)**3)/(msmbh))
        # Calculating and adding oscillatory signal
        sinus=A*np.sin(2*np.pi*tt/(P*365))
        ss = ss + sinus
        yy = np.zeros(times)
        for i in range(times):
            # Adding error and noise to each magnitude value
            yy[i] = ss[i] + np.random.normal(0,((noise*ss[i])),1) + np.sqrt(err[i])
    
        return tt, yy
    
    # Final light curve without oscillations
    if oscillations == False:
        yy = np.zeros(times)
        for i in range(times):
            # Adding error and noise to each magnitude value
            yy[i] = ss[i] + np.random.normal(0,((noise*ss[i])),1) + np.sqrt(err[i])
    
        return tt, yy



def LC_opsim(mjd,t,y):
    """ 
    Returns a hypothetical light curve sampled in a given OpSim strategy.
    User needs to provide a reference light curve for sampling (usually a continous
    light curve with 1 day cadence, see LC_conti() function).
    
    Parameters:
    -----------
    mjd: np.array
        Modified Julian Date obtained from OpSim. It is the time of each sampling of the light curve
        during the LSST operation period in one of the filters and specified sky coordinates.
    t: np.array
        Days during the survey on which we had an observation for a continuous reference light curve.
    y: np.array
        Light curve magnitudes for continuous reference light curve.
     
    Returns:
    --------
    top: np.array
        Days during the survey when we had an observation (sampling) in a given OpSim strategy.
    yop: np.array
        Light curve magnitude taken from the reference light curve on days we had an observation 
        (sampling) in a given OpSim strategy.
    """
    
    # Convert MJD to survey days
    top=np.ceil(mjd-mjd.min())
    
    # Reference light curve sampling
    yop=[]
    for i in range(len(top)):
        abs_vals = np.abs(t-top[i])
        
        # Find matching days and their index
        bool_arr = abs_vals < 1
        if bool_arr.sum() != 0:
            index = (np.where(bool_arr)[0])[0]
            yop.append(y[index])
            
        # Case when we don't have a match
        elif bool_arr.sum() == 0:
            yop.append(-999)
    yop=np.asarray(yop)
    
    # Drop placeholder values (-999)
    top = top[yop!=-999]
    yop = yop[yop!=-999]
    
    return top,yop
  

def sf(t,y,z=0):
    """
    Calculates the structure function (SF) parameters using the first-order SF method.
    
    Parameters:
    -----------
    t: np.array
        Days when we had an observation (sampling).
    y: np.array
        Light curve magnitudes.
    z: float, default=0
        Redshift.

    Returns:
    --------
    s: np.array
        Mean of the squared flux difference between consecutive light curve points in bins with edges 
        defined by y. Used for plotting the y-axis of the structure function visualization.
    edges: np.array
        Bin edges for the range of intervals between consecutive observation times (time scales). 
        Used for plotting the x-axis in the structure function visualization.
    """
    
    dtr = []
    dyr = []
    obs = np.asarray(y.shape)[0]
    for i in range(obs-1):
        dtr.append(t[i+1:obs] - t[i])
        dyr.append((y[i+1:obs] - y[i])*(y[i+1:obs] - y[i]))

    dtr = np.concatenate(dtr, axis=0)
    dyr = np.concatenate(dyr, axis=0)
    
    s, edges, _ = binned_statistic(dtr/(1+z), dyr, statistic='mean', bins=np.logspace(0,4,100))
    
    return s, edges
    
    
def LC_SF_viz(T, deltatc, opsims, labels, oscillations=True, A=0.14, noise=0.00005, z=0, 
              frame='observed'):
    """ 
    Calculate and plot the continuous reference light curve, hypothetical light curves from
    an arbitrary number of OpSim strategies and their structure functions.
    
    Parameters:
    -----------
    T: int
        Total time span of the continuous reference light curve.
    deltatc: int, default=1
        Cadence (or sampling rate) - time interval between two consecutive samplings of the light 
        curve in days.
    opsims: list of arrays
        list with arbitrary number of arrays containing OpSim light curve sampling times in the
        form of Modified Julian Date.
    labels: list of strings
        list containing the labels for plots of each OpSim light curve.
    oscillations: bool, default=True
        If True, light curve simulation will take an oscillatory signal into account.
    A: float or int, default=0.14
        Amplitude of the oscillatory signal in magnitudes (used only if oscillations=True).
    noise: float, default=0.00005
        Amount of noise to include in the light curve simulation.
    z: float, default=0
        Redshift.
    frame: {'observed', 'rest'}, default='observed'
        Frame of reference.
    """
    # Every time the LC_conti function is called, a random simulated light curve is generated. 
    # We do this only once because we want to evaluate OpSim light curves on the same referent 
    # continuous light curve.
    tt, yy = LC_conti(T, deltatc, oscillations, A, noise, z, frame)
    
    # Calculate structure function for the continuous light curve
    sc, edgesc = sf(tt, yy, z=z)
    
    # Create a figure where all the light curves will be stored
    fig1 = plt.figure(figsize=(15,10))
    
    # Continuous light curve is plotted first
    ax11 = fig1.add_subplot(len(opsims)+1,1,1)
    ax11.plot(tt, yy, 'ko', markersize = 1, label='1 day cadence')
    plt.setp(ax11.get_xticklabels(), visible=False)
    custom_xlim = (0, T)
    custom_ylim = (yy.min()-0.3, yy.max()+0.3)
    ax11.set_ylabel('magnitude', fontsize = 14)
    ax11.tick_params(direction='in', pad = 5)
    plt.setp(ax11, xlim=custom_xlim, ylim=custom_ylim)
    ax11.legend(loc='upper right')
    ax11.grid(True)
    
    # Lists to store the results from each OpSim light curve and later use those
    # to calculate their structure functions.
    tops = []
    yops = []
    
    # Plot all OpSim light curves provided by user
    for i, mjd in enumerate(opsims):
        top, yop = LC_opsim(mjd, tt, yy)
        tops.append(top)
        yops.append(yop)
        
        ax = fig1.add_subplot(len(opsims)+1,1,i+2)
        ax.plot(top, yop, 'ko', markersize = 1, label='%s' %labels[i])
        
        if (i+1) == len(opsims):
            ax.set_xlabel('t [days]', fontsize=14)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
    
        ax.set_ylabel('magnitude', fontsize = 14)
        ax.tick_params(direction='in', pad = 5)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        ax.legend()
        ax.grid(True)    
    
    tops = np.asarray(tops, dtype=object)
    yops = np.asarray(yops, dtype=object)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.05)
    fig1.suptitle('Light curves', y=0.92, fontsize=17)
    
    plt.show()
    
    
    # We create a second figure to store structure function plots.
    fig2 = plt.figure(figsize=(15,6))
    # Plot the structure function of the continuous light curve.
    ax21 = fig2.add_subplot(121)
    ax21.plot(np.log10(edgesc[:-1]+np.diff(edgesc)/2), np.log10(np.sqrt(sc)), 'ko-',linewidth=1, markersize=3,label='1 day cadence')
    ax21.set_xlabel(r'$\log_{10}(\Delta t)$', fontsize = 14)
    ax21.set_ylabel(r'$\log_{10} \ SF$', fontsize = 14)
    ax22 = fig2.add_subplot(122)
    ax22.plot(np.log10(edgesc[:-1]+np.diff(edgesc)/2), np.sqrt(sc), 'ko-',linewidth=1, markersize=3,label='1 day cadence')
    ax22.set_xlabel(r'$\log_{10}(\Delta t)$', fontsize = 14)
    ax22.set_ylabel('SF', fontsize = 14)
    
    # Plot the structure functions of OpSim light curves over the structure function
    # of the continuous light curve.
    color=iter(plt.cm.cool(np.linspace(0,1,len(opsims))))
    i = 0 # counter
    for topp, yopp in zip(tops, yops):
        s, edge = sf(topp, yopp, z=z)
        c=next(color)
        ax21.plot(np.log10(edge[:-1]+np.diff(edge)/2), np.log10(np.sqrt(s)), c=c, linewidth=1, marker='o', markersize=3, label='%s'%labels[i])
        ax22.plot(np.log10(edge[:-1]+np.diff(edge)/2), np.sqrt(s), c=c, linewidth=1, marker='o', markersize=3, label='%s'%labels[i])
        i = i+1
    
    axs = [ax21, ax22]
    
    for ax in axs:
        ax.tick_params(direction='in', pad = 5)
        ax.legend()
        ax.grid(True)
        
    fig2.suptitle('Structure functions', y=0.96, fontsize=17)
    
    
def var_cad(tt, yy, years=10, 
            m1=0,m2=0,m3=0,m4=0,m5=0,m6=0,m7=0,m8=0,m9=0,m10=0,m11=0,m12=0):
    """
    Return light curves with user defined monthly cadences. If possible, use light curves with 
    1 day cadence for the entire duration of the survey.
    
    Parameters:
    -----------
    tt: np.array
        Days during the survey on which we had an observation for a continuous reference light curve.
    yy: np.array
        Magnitudes of the reference light curve.
    years: int
        Duration of the survey in years.
    m1; m2; ... ; m12: int in range [0,30], default=0 (no obseravtions)
        Cadence for each month in a year of the survey. The same combination of monthly cadences 
        is used for each year of the survey.
        
    Returns:
    --------
    tm: np.array
        Days during the survey on which we had an observation for resulting light curve with 
        user defined variable cadence.
    ym: np.array
        Magnitudes for the resulting light curve with user defined variable cadence.
    """

    # Monthly cadances
    month_cad = [m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12]
    years = int(years)
    
    # Define months in terms of survey days for the first year
    bounds = [(0,31),(31,59),(59,90),(90,120),(120,151),(151,181),(181,212),(212,243),(243,273),(273,304),(304,334),(334,365)]
    
    # Lists where days with observation for each month will be stored
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    l10 = []
    l11 = []
    l12 = []
    
    ls = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]
    
    # Calculating days with observations for the whole survey duration (10 yrears) 
    # taking into account variable cadences.
    for mc, b, l in zip(month_cad, bounds, ls):
        if mc != 0:  # case when there are observations for a given month
            # First year
            lb, ub = b
            tm = np.arange(lb, ub, int(mc))
            l.append(tm)
            # Years 2-10
            for yr in range(years-1):
                tm = tm + 365
                l.append(tm)
            l = np.asarray(l)
        
        elif mc == 0:
            for yr in range(years):
                l.append(np.zeros(30))
            l = np.asarray(l)
            
    tm = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12), axis=1).flatten()
    tm = tm[tm != 0]
    
    # Evaluating light curve flux for days obtained using variable cadences
    ym=[]
    for i in range(len(tm)):
        abs_vals = np.abs(tt-tm[i])
        bool_arr = abs_vals < 1
        if bool_arr.sum() != 0:
            index = (np.where(bool_arr)[0])[0]
            ym.append(yy[index])
        elif bool_arr == 0:
            ym.append(-999)
    ym=np.asarray(ym)
    tm = tm[ym!=-999]
    ym = ym[ym!=-999]
    
    return tm, ym



def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    
def SF_heatmap(mjd, label, lb='map', nlc=50, frame='observed', save=True, cmap='RdBu_r', c=30):
    """
    Parameters:
    -----------
    mjd: np.array
        Days with observations in a given OpSim realization (in MJD format).
    label: str
        Label indicating which OpSim is used. This will appear in the visualization.
    nlc: int, default=50
        Number of artificial light curves generated for each redshift bin.
    frame: {'observed', 'rest'}, default='observed'
        Frame of reference.
    cmap: str, default='RdBu_r'
        Colormap for filled contour plot.
    c: int, defalt=30
        Determines the number and positions of the contour lines (regions).
    save: bool, default=True
        Choose whether you want to save the obtained map.
    """
    # Define redshift bins
    zbin = np.linspace(0.5,7.5,8)
    zbin = np.insert(zbin,0,0)
    
    # Converting MJD to survey days
    T=np.int(mjd.max()-mjd.min()+1)
    swop=[]
    wedgeop=[]
    scop=[]
    edgecop=[]
    i=0

    total = len(zbin)*(nlc);
    progress = 0;
    
    # We generate a number (nlc) of light curves for each redshift bin
    for z in zbin:
        for w in range(nlc):
            # Generating continuous light curve (cadence=1d)
            tt, yy = LC_conti(T, z=z, frame=frame)
            sn, edgesn = sf(tt,yy,z=z)
            # Calculating SF for the current continuous light curve
            scop.append(sn)
            edgecop.append(edgesn)
            # Generating OpSim light curve evaluated on the current continuous light curve
            top,yop=LC_opsim(mjd,tt,yy)
            # Calculating SF for the current OpSim light curve
            srol,edgesrol=sf(top,yop,z=z)
            swop.append(srol)
            wedgeop.append(edgesrol)
            
            progressBar(progress, total);
            progress = progress + 1;
        i=i+1  # counter

 
    swop=np.asarray(swop)
    swop=swop.reshape(9,nlc,99)
    scop=np.asarray(scop)
    scop=scop.reshape(9,nlc,99)
    razrol=[]
    for z in range(9):
        for r in range(nlc):
            # Calculating the SF metric
            razrol.append((np.nan_to_num(np.sqrt(scop[z,r,:]))-np.nan_to_num(np.sqrt(swop[z,r,:]))))
    
    razrol9=np.asarray(razrol)
    razrol9=razrol9.reshape(9,nlc,99)
    # We take the mean of generated light curves for each redshift bin.
    raz2=np.nanmean(razrol9[:,:,:],axis=1)
    
    # Plotting the countur plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(np.log10(edgesn[:-1])+((np.log10(edgesn[1])-np.log10(edgesn[0]))/2), zbin)
    sf_max = raz2.max()
    sf_min = raz2.min()
    if ((sf_max > abs(sf_min)) & (sf_max >= 0) & (sf_min < 0)):
        im = plt.contourf(X, Y, raz2, c, cmap=cmap, vmax=sf_max, vmin=-sf_max)
    elif ((sf_max < abs(sf_min)) & (sf_max >= 0) & (sf_min < 0)):
        im = plt.contourf(X, Y, raz2, c, cmap=cmap, vmax=abs(sf_min), vmin=sf_min)
    else:
        im = plt.contourf(X, Y, raz2, c, cmap=cmap, vmax=sf_max, vmin=sf_min)
    ax.set_xlabel(r'$\mathrm{log_{10}}(\Delta \mathrm{t})$',fontsize=16)
    ax.set_ylabel(r'z',fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlim(0,4)
    ax.set_ylim(0,7)
    cbar=fig.colorbar(im)
    cbar.set_label('averaged SF(1 day cad.) - SF(%s)' %(label), fontsize=13)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='both', which='major', labelsize=15, direction='out', length = 5, pad = 5)
    if save==True:
        plt.savefig(lb+'.pdf', dpi=250)
    plt.show()

def LC_plot(tt, yy, T):
    """
    Simple plotting function.
    
    Parameters:
    -----------
    tt: np.array
        Days when the light curve was sampled.
    yy: np.array
        Light curve magnitudes.
    T: int
        Total time span of the light curve. 
    """
    
    fig = plt.figure(figsize=(15,5))
    
    ax = fig.add_subplot(111)
    ax.plot(tt, yy, 'ko', markersize = 1, label='1 day cadence')

    custom_xlim = (0, T)
    custom_ylim = (yy.min()-0.1, yy.max()+0.1)
    ax.set_xlabel('t [days]', fontsize = 18, labelpad=10)
    ax.set_ylabel('magnitude', fontsize = 18, labelpad=10)
    ax.tick_params(direction='in', pad = 5, labelsize=13)
    plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    ax.legend(fontsize=15)
    ax.grid(True)

# get opsim cadence file
def getOpSimCadence(opsim, name, ra = 0, dec = 0, fil = 'r'):
    
    
    import lsst.sims.maf.metrics as metrics
    import lsst.sims.maf.db as db
    import lsst.sims.maf.slicers as slicers

    import lsst.sims.maf.metricBundles as mb


    
    # We are only interested in date of observation for chosen filter and field point
    colmn = 'observationStartMJD';
    
    # Directory where tmp files are going to be stored
    outDir = 'TmpDir'
    resultsDb = db.ResultsDb(outDir=outDir)
    
    # Using MAF PassMetrics to get all desired data from the chosen opsim realisation
    metric=metrics.PassMetric(cols=[colmn,'fiveSigmaDepth', 'filter'])
    
    # Select RA and DEC
    slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
    
    # Add sql constraint to select desired filter
    sqlconstraint = 'filter = \'' + fil + '\''
    
    
    # Run simulation
    bundle = mb.MetricBundle(
        metric, slicer, sqlconstraint, runName=name)
    bgroup = mb.MetricBundleGroup(
        {0: bundle}, opsim, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll();
    filters = np.unique(bundle.metricValues[0]['filter'])
    mv = bundle.metricValues[0]
    
    
    # Get dates
    mjd = len(mv[colmn])
    
    # Out file
    filepath = name + "_" + str(ra) + "_" + str(dec) + "_" + fil +".dat";
    with open(filepath, 'w') as file_handler:
        for item in mv[colmn]:
            file_handler.write("{}\n".format(item))

    mjd=np.loadtxt(filepath)
    mjd=np.sort(mjd)
    
    return mjd;


