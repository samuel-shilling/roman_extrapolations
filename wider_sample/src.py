# import logging
# logging.getLogger('redback').setLevel(logging.ERROR)
# logging.getLogger('bilby').setLevel(logging.ERROR)
import redback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import dill
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import afterglowpy as grb
import bilby
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import swifttools.ukssdc.data as ud #https://www.swift.ac.uk/API/ukssdc/data/GRB.md
import swifttools.ukssdc.query as uq #add link
import swifttools.ukssdc.xrt_prods as ux #add link
import swifttools.ukssdc.data.GRB as udg # https://www.swift.ac.uk/API/ukssdc/data/GRB.md
from swifttools.ukssdc import plotLightCurve #https://www.swift.ac.uk/API/ukssdc/commonFunc.md#plotlightcurve
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from tqdm import tqdm
from uncertainties import unumpy
from uncertainties import ufloat
import sys
num_cpus = os.cpu_count()
num_cpus=num_cpus+2 #ensure saturation
import tqdm
import tqdm.auto
def tqdm_plain(*args, **kwargs):
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return tqdm.std.tqdm(*args, **kwargs)
tqdm.auto.tqdm = tqdm_plain 
mpl.rcParams.update(mpl.rcParamsDefault) #fix a LaTeX bug with matplotlib in redback
mpl.rcParams['text.usetex'] = False

#set the root 
root = Path(__file__).resolve().parent

#######This boolean controls whether we want to sample or not
sample=False


#######
#######
####### Section 1. Retrieving data and processing for Redback compatibility
#######
#######


# We need to read in the light curve data to get the time of the final data point 
Mega_dataframe=pd.read_csv('{}/central_data_object_processed_and_selected.csv'.format(root),encoding='unicode_escape')

# Do the unit conversions to Redback formats
Mega_dataframe['Flux_mjy']=Mega_dataframe['Flux density'].values*1e26 #convert from erg/s/cm2/hz to mJy
Mega_dataframe['Flux_err_mjy']=Mega_dataframe['Flux density error (symmetric)'].values*1e26 #convert from erg/s/cm2/hz to mJy
Mega_dataframe['Time_d']=Mega_dataframe['Time']/(60*60*24)
Mega_dataframe['TimeErr']=(np.sqrt(np.power(Mega_dataframe['TimePos'],2))+np.sqrt(np.power(Mega_dataframe['TimeNeg'],2)))/2
Mega_dataframe['Time_e']=Mega_dataframe['TimeErr'].values/(60*60*24) #convert from s to d

# I have already modelled full sample in a previous project so we can skip straight to reading in the results object

#######
#######
####### Section 2. Setup sampling settings, and data object, and then sample
#######
#######


# Define model wrappers (since we are using a customised version of the Redback model)
class convert_thvthc_to_thv(object):

    def __init__(self, thvthc, thc):
        self.thvthc = thvthc
        self.thc = thc
        self.thv = self.calculate_thv

    @property
    def calculate_thv(self):
        thv=(self.thvthc)*(self.thc)
        return thv
    
#the wrapped models (this time we use all jet structures)
def wrapped_tophat(time,**kwargs):
    #convert thvthc into thv before passing into the model
    kwargs['thv']=convert_thvthc_to_thv(kwargs['thvthc'],kwargs['thc']).thv
    #call model
    y=redback.transient_models.afterglow_models.tophat(time,**kwargs)
    return(y)
def wrapped_gauss(time,**kwargs):
    #convert thvthc into thv before passing into the model
    kwargs['thv']=convert_thvthc_to_thv(kwargs['thvthc'],kwargs['thc']).thv
    #call model
    y=redback.transient_models.afterglow_models.gaussian(time,**kwargs)
    return(y)
def wrapped_SPL(time,**kwargs):
    #convert thvthc into thv before passing into the model
    kwargs['thv']=convert_thvthc_to_thv(kwargs['thvthc'],kwargs['thc']).thv
    #call model
    y=redback.transient_models.afterglow_models.smoothpowerlaw(time,**kwargs)
    return(y)

#make a dictionary to convert the full model name to the wrapped model name
translater={'tophat':wrapped_tophat,
            'gaussian':wrapped_gauss,
            'smoothpowerlaw':wrapped_SPL}

            
#read in Redback results
with open("{}/results_merged_x_rebin_full_v2.pkl".format(root), "rb") as f:
    full_results_dictionary = dill.load(f)

#get the best fit model for each GRB stored in a dictionary (we need to call this as wrapped_model later on)
log_bayes=dict()
wrapped_model_dictionary=dict()
for name in full_results_dictionary.keys():
    if name!='Repeat':
        for each_model in ['tophat','gaussian','smoothpowerlaw']:
            log_bayes[each_model]=full_results_dictionary[name][each_model].log_10_bayes_factor
        best_model=max(log_bayes,key=log_bayes.get) #best fit model 
        wrapped_model_dictionary[name]=best_model


#######
#######
####### Section 3. Results manipulation (using in-built Redback tools)
#######
#######



#No need for this bit which was just here for demo purposes for 260924A
#Let's just skip straight to the main results section below



#######
#######
####### Section 4. Results manipulation (extracting the results from Redback and doing manually)
#######
#######


#Get the Rubin frequency and sensitivity (see bands at: https://roman.gsfc.nasa.gov/images/pdf/Roman-Instruments-Chart.pdf) 
#^^ so we can extrpolate the  model posterior to this band and overlay the sensitivity
frequencies={'XRT': 2.41799050402417e17,
             'W146_Hz':3e8/( ((0.93+2)/2) *1e-6),
             'R062_Hz':3e8/( ((0.48+0.76)/2) *1e-6)}
limiting_mags={'XRT': None,
                'W146_Hz':28.3,
                'R062_Hz':28.5}


#initialize dictionary to store results in for each named GRB
time_until_fade={}

for name in wrapped_model_dictionary.keys():

    #initialise results entry
    time_until_fade[name]={}

    #get the actual light curve data (observed) and convert from s to d
    data=Mega_dataframe[Mega_dataframe['GRB']==name]

    #get the best model
    best_model=wrapped_model_dictionary[name]

    #get the wrapped model name
    wrapped_model=translater[best_model]

    #get the redback results object straight from the dictionary
    results_dictionary=full_results_dictionary[name][best_model]

    #get the posterior
    posterior=results_dictionary.posterior.copy()

    #set time interval to produce Rubin light curve over; lets go between 0.01 days and 400 days (evaluated in log space)
    t_d=np.power(10,np.linspace(np.log10(data['Time_d'].min()),np.log10(10000),200)) 

    #get the max likelihood values
    posterior.sort_values(by='log_likelihood', inplace=True)
    manual_kwargs=dict(posterior[posterior.columns].iloc[-1])

    #set the frequency to Rubin and output format to magnitude
    #simulate the light curve in each band and calculate the time since t0 before the flux drops below limiting mag
    for band in frequencies.keys():
        if band!='XRT':
            manual_kwargs['frequency']=frequencies[band]
            manual_kwargs['output_format']='magnitude'

            #call the model and simulate the data
            y=wrapped_model(t_d,**manual_kwargs)

            #calculate time since t0 until the flux goes below the limiting magnitude
            mask=(y<=limiting_mags[band])
            if not np.any(mask):
                time=0.0  #Never detectable
            else:
                time = t_d[mask]
                time=time[-1]
            time_until_fade[name][band]=round(time,2)

    #make the plots to visualise all this for just a couple of GRBs
    if name in ['050126','060729','130427A']:

        fig,axs=plt.subplots(nrows=len(frequencies.keys()),ncols=1,figsize=(8, 4 * len(frequencies.keys())))
                             
        #plot in each band
        for count,band in enumerate(frequencies.keys()):

            #setup keyword arguments
            manual_kwargs['frequency']=frequencies[band]
            if band!='XRT':
                manual_kwargs['output_format']='magnitude'
            else:
                manual_kwargs['output_format']='flux_density'

            #simulate
            y=wrapped_model(t_d,**manual_kwargs)

            #OPTIONAL: manually evaluate the credible intervals
            integers = np.arange(len(posterior)) #get all posterior elements
            indices = np.random.choice(integers, size=100) 
            rand_kwargs=[posterior.iloc[x] for x in indices] #take a row of the posterior for each random element
            for zz in rand_kwargs: #need to add the settings and freq as these arent in the posterior
                zz['frequency']=manual_kwargs['frequency']
                zz['output_format']=manual_kwargs['output_format']
            random_ys=[wrapped_model(t_d, **random_params) for random_params in rand_kwargs] #get random y values at each time for each set of random kwargs from the posterior
            lower_bound, upper_bound, _ = redback.utils.calc_credible_intervals(samples=random_ys, interval=0.68) #calculate the level (68%) credible intervals between each sample


            ###plot the results with the limiting magnitude and the final XRT observation
            axs[count].plot(t_d,y,label='XRT fit extrapolated' if band!='XRT' else 'XRT fit',linestyle='-',color='red',lw=2) #plot the model data itself
            axs[count].fill_between(t_d, lower_bound, upper_bound,alpha=0.25, color='red') #plot credible intervals
            if band!='XRT':
                axs[count].axhline(limiting_mags[band],c='black',label='Roman sensitivity')
                axs[count].axvline(time_until_fade[name][band],c='black',ls=':',label='{} days until not detectable'.format(time_until_fade[name][band]))
                
            else:
                axs[count].errorbar(x=data['Time_d'],xerr=data['Time_e'],y=data['Flux_mjy'],yerr=data['Flux_err_mjy'],fmt='o',color='black',ms=2,fillstyle=None,label='Observed XRT data')
            axs[count].axvline(data['Time_d'].max(),c='orange',label='Time of last swift/XRT observation')
            axs[count].axvline(365,c='green',label='1 year since explosion')

            #format
            axs[count].set_xscale('log')
            if band!='XRT':
                axs[count].yaxis.set_inverted(True)
            else:
                axs[count].set_yscale('log')
            axs[count].tick_params(axis='both', which='major', labelsize=16)
            axs[count].set_ylabel('Magnitude' if band!='XRT' else 'Flux density at 1 keV (mJy)',fontsize=16)
            axs[count].set_xlabel('Time since explosion [d]',fontsize=16)
            axs[count].set_title('GRB {}, {} fit, {} band'.format(name,best_model,band))
            axs[count].legend()
        plt.tight_layout()
        plt.savefig('{}/figures/{}.pdf'.format(root,name),dpi=200)
        plt.show()

#save the results
with open("{}/figures/results_detectable_times.pkl".format(root), "wb") as f:
    dill.dump(time_until_fade, f)


#Histogram the time until not detectable
frequencies_for_histogram=frequencies.copy()
frequencies_for_histogram.pop('XRT')
fig,axs=plt.subplots(nrows=len(frequencies_for_histogram.keys()),ncols=1,figsize=(8, 4 * len(frequencies_for_histogram.keys())))    
for count,band in enumerate(frequencies_for_histogram.keys()):
    times=[results[band] for results in time_until_fade.values()]
    bins=np.linspace(np.min(times),np.max(times),num=100)
    counts,bins=np.histogram(times,bins=bins)
    axs[count].stairs(counts,bins,color='black',label='{} (median={} d)'.format(band,round(np.median(times),2)),lw=2,zorder=100)
    axs[count].set_ylabel('Counts')
    axs[count].set_xlabel('Time since T0 until not detectable [days]')
    axs[count].legend()
    print(band, round(np.median(times),2))
plt.savefig('{}/figures/hist.pdf'.format(root),dpi=200)
plt.show()
