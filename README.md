## The effects of cadence selection for determination of time lags, oscillations and quality of Structure Function (SF) 
[![HitCount](http://hits.dwyl.com/LSST-sersag/agn_cadences.svg)](http://hits.dwyl.com/LSST-sersag/agn_cadences)


We intent to explore the effects of cadences required for determination of time lags, oscillations and quality of structure function (SF). 

For time-lags and oscillation detection, we design a multiple regression model of the uncertainty of time-lag and oscillations from the existing data-set of decade-long reverberation mapping campaigns of eight type 1 AGN of different variability and optical spectral characteristics. For the SF analysis, we generate a suit of 10-year long light curves containing oscillatory signal, for which we select different sampling strategies: ideal case with homogeneous 1-day cadence; set of "gappy" light curves with unobserved gaps of different length; several LSST Operations Simulation realizations with different cadence strategies.

The discussed multiple regression model for cadence prediction can help in designing both spectroscopic and photometric surveys. We showed that the reconstruction of SF properties of AGN light-curves with oscillatory signals is strongly dependant on the observing cadences, which could be important for the selection of operation strategy for upcoming large photometric and spectroscopic surveys. 

### Requirements

1. Requred python libreries:

* `numpy`
* `matplotlib`
* `pandas`
* `healpy`
* `scipy`
* `ipywidgets`
* `lsst.sims.maf` (see https://github.com/lsst/sims_maf)

2. For the purpose of testing OpSim cadences it is obligatory to download desired OpSim realisation ([OpSim LSST](https://www.lsst.org/scientists/simulations/opsim), [OpSim realisations](https://epyc.astro.washington.edu/~lynnej/opsim_downloads/) ). In the code examples we used FBS  1.7 OpSim realisation.

### Repo organization
 
* Folder `Examples` contains README.md file and Jupyter notbooks with detailed explanations.
* Folder  `Cadences` contains main code

### References

A. Kovačević, D. Ilić, L. Č. Popović, V. Radović, I. Jankov, I. Yoon, I. Čvorović-Hajdinjak, S. Simić [On possible proxies of AGN light curves cadence selection in future time domain surveys](#) (2021) <i><u> in prepration </i></u>
