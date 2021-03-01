# lsst_agn_sf



Motivated by upcoming large photometric and spectroscopic surveys covering a broad range of wavelengths and regularly monitoring a large fraction of the sky, such as the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) or the Manuakea Spectroscopic Explorer (MSE), we aim to conceptualize the statistical proxies which will explore the effects of cadences required for determination of time lags, oscillations and quality of structure function (SF).

 For time-lags and oscillation detection, we design a multiple regression model of the uncertainty of time-lag and oscillations from the existing data-set of decade-long reverberation mapping campaigns of eight type 1 AGN of different variability and optical spectral characteristics. For the SF analysis, we generate a suit of 10-year long light curves containing oscillatory signal, for which we select different sampling strategies: ideal case with homogeneous 1-day cadence; set of "gappy" light curves with unobserved gaps of different length; several LSST Operations Simulation realizations with different cadence strategies.
 
 
 The discussed multiple regression model for cadence prediction can help in designing both spectroscopic and photometric surveys. We showed that the reconstruction of SF properties of AGN light-curves with oscillatory signals is strongly dependant on the observing cadences, which could be important for the selection of operation strategy for surveys such as the LSST. The proposed multiple regression is promising for predicting AGN observables cadences in both LSST-like and sparser strategies, but it will be tested further in larger samples of objects such as SDSS.

In Jupyter file cadences_file.ipynb we have shown results obtained using the lastest LSST OpSim realisation 1.7.
