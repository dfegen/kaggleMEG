# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:23:29 2014

@author: Drew
"""

import scipy.io
import numpy as np
import mne

###############################################################################

#MNE has three main datatypes or objects:
    #Raw
    #Epoch
    #Evoked

#we do not have the Raw data, only provided the Epoch data in .mat format
#   which we then have to convert into MNE Epoch object (code below) 
#once an Epoch object, can then convert into Evoked object (code below)


###############################################################################

#load in the .mat file with scipy
train_subject01 = scipy.io.loadmat('/home/tobin/kaggle/MEG_decode/data/train_subject01.mat')

#above command returns a dict
list(train_subject01.keys())

#'X' is the data, 'y' is the label (for training data)
print train_subject01['X'].shape
print train_subject01['y'].shape

#for MNE need to create 'events' - each ROW is a trial, each COLUMN is:
    #column1: time when event occurred
    #column2: value of trigger channel just before the change and now, usually 0
    #column3: condition
time=np.array([0]*len(train_subject01['y']))
trigger_value=np.array([0]*len(train_subject01['y']))
condition=np.squeeze(np.array(train_subject01['y']))

events=np.column_stack((time,trigger_value,condition))


#use mne.fiff.array.create_info() to make an info object
channel_names=['MEG_0113','MEG_0112','MEG_0111','MEG_0122','MEG_0123','MEG_0121','MEG_0132','MEG_0133','MEG_0131','MEG_0143','MEG_0142','MEG_0141','MEG_0213','MEG_0212','MEG_0211','MEG_0222','MEG_0223','MEG_0221','MEG_0232','MEG_0233','MEG_0231','MEG_0243','MEG_0242','MEG_0241','MEG_0313','MEG_0312','MEG_0311','MEG_0322','MEG_0323','MEG_0321','MEG_0333','MEG_0332','MEG_0331','MEG_0343','MEG_0342','MEG_0341','MEG_0413','MEG_0412','MEG_0411','MEG_0422','MEG_0423','MEG_0421','MEG_0432','MEG_0433','MEG_0431','MEG_0443','MEG_0442','MEG_0441','MEG_0513','MEG_0512','MEG_0511','MEG_0523','MEG_0522','MEG_0521','MEG_0532','MEG_0533','MEG_0531','MEG_0542','MEG_0543','MEG_0541','MEG_0613','MEG_0612','MEG_0611','MEG_0622','MEG_0623','MEG_0621','MEG_0633','MEG_0632','MEG_0631','MEG_0642','MEG_0643','MEG_0641','MEG_0713','MEG_0712','MEG_0711','MEG_0723','MEG_0722','MEG_0721','MEG_0733','MEG_0732','MEG_0731','MEG_0743','MEG_0742','MEG_0741','MEG_0813','MEG_0812','MEG_0811','MEG_0822','MEG_0823','MEG_0821','MEG_0913','MEG_0912','MEG_0911','MEG_0923','MEG_0922','MEG_0921','MEG_0932','MEG_0933','MEG_0931','MEG_0942','MEG_0943','MEG_0941','MEG_1013','MEG_1012','MEG_1011','MEG_1023','MEG_1022','MEG_1021','MEG_1032','MEG_1033','MEG_1031','MEG_1043','MEG_1042','MEG_1041','MEG_1112','MEG_1113','MEG_1111','MEG_1123','MEG_1122','MEG_1121','MEG_1133','MEG_1132','MEG_1131','MEG_1142','MEG_1143','MEG_1141','MEG_1213','MEG_1212','MEG_1211','MEG_1223','MEG_1222','MEG_1221','MEG_1232','MEG_1233','MEG_1231','MEG_1243','MEG_1242','MEG_1241','MEG_1312','MEG_1313','MEG_1311','MEG_1323','MEG_1322','MEG_1321','MEG_1333','MEG_1332','MEG_1331','MEG_1342','MEG_1343','MEG_1341','MEG_1412','MEG_1413','MEG_1411','MEG_1423','MEG_1422','MEG_1421','MEG_1433','MEG_1432','MEG_1431','MEG_1442','MEG_1443','MEG_1441','MEG_1512','MEG_1513','MEG_1511','MEG_1522','MEG_1523','MEG_1521','MEG_1533','MEG_1532','MEG_1531','MEG_1543','MEG_1542','MEG_1541','MEG_1613','MEG_1612','MEG_1611','MEG_1622','MEG_1623','MEG_1621','MEG_1632','MEG_1633','MEG_1631','MEG_1643','MEG_1642','MEG_1641','MEG_1713','MEG_1712','MEG_1711','MEG_1722','MEG_1723','MEG_1721','MEG_1732','MEG_1733','MEG_1731','MEG_1743','MEG_1742','MEG_1741','MEG_1813','MEG_1812','MEG_1811','MEG_1822','MEG_1823','MEG_1821','MEG_1832','MEG_1833','MEG_1831','MEG_1843','MEG_1842','MEG_1841','MEG_1912','MEG_1913','MEG_1911','MEG_1923','MEG_1922','MEG_1921','MEG_1932','MEG_1933','MEG_1931','MEG_1943','MEG_1942','MEG_1941','MEG_2013','MEG_2012','MEG_2011','MEG_2023','MEG_2022','MEG_2021','MEG_2032','MEG_2033','MEG_2031','MEG_2042','MEG_2043','MEG_2041','MEG_2113','MEG_2112','MEG_2111','MEG_2122','MEG_2123','MEG_2121','MEG_2133','MEG_2132','MEG_2131','MEG_2143','MEG_2142','MEG_2141','MEG_2212','MEG_2213','MEG_2211','MEG_2223','MEG_2222','MEG_2221','MEG_2233','MEG_2232','MEG_2231','MEG_2242','MEG_2243','MEG_2241','MEG_2312','MEG_2313','MEG_2311','MEG_2323','MEG_2322','MEG_2321','MEG_2332','MEG_2333','MEG_2331','MEG_2343','MEG_2342','MEG_2341','MEG_2412','MEG_2413','MEG_2411','MEG_2423','MEG_2422','MEG_2421','MEG_2433','MEG_2432','MEG_2431','MEG_2442','MEG_2443','MEG_2441','MEG_2512','MEG_2513','MEG_2511','MEG_2522','MEG_2523','MEG_2521','MEG_2533','MEG_2532','MEG_2531','MEG_2543','MEG_2542','MEG_2541','MEG_2612','MEG_2613','MEG_2611','MEG_2623','MEG_2622','MEG_2621','MEG_2633','MEG_2632','MEG_2631','MEG_2642','MEG_2643','MEG_2641']
channel_types=['grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag','grad','grad','mag']
input_info=mne.fiff.array.create_info(ch_names=channel_names,sfreq=250,ch_types=channel_types)


#function to read in .mat epoch data into a MNE Epoch object, creates an Epoch object 'epochs'
epochs=mne.epochs.EpochsArray(data=train_subject01['X'],
                            info=input_info,
                            events=events,
                            tmin=-0.5,
                            event_id=dict(scramble=0,face=1))

print epochs
print epochs.info

dir(mne.epochs)


###############################################################################
# Various manipulations to do with Epoch objects

#indicate bad channels
epochs.info['bads'] += ['MEG_0113', 'MEG_0112']
#indicate some more bad channels (original bad channels + new one)
epochs.info['bads'] += ['MEG_0111']
print epochs.info



#pick just the magnetometers or gradiometers channels
mag_picks = mne.epochs.pick_types(epochs.info, meg='mag', exclude='bads')
grad_picks = mne.epochs.pick_types(epochs.info, meg='grad', exclude='bads')
#--> many Epochs methods have a 'pick' input which is the channels you wish to include
#       this can be input for that arg


#save epoch data as .fif
epochs.save('train_subject01_rawEpochs.fif')

#read in saved epoch data
saved_epochs = mne.read_epochs('/home/tobin/Desktop/train_subject01_rawEpochs.fif')
#--> no option to 'baseline' the data when reading in as Epoch
#    if save as Evoked and then read in as Evoked, can 'baseline' [mne.fiff.read_evoked()]


max_face_Epoch = [e.max() for e in epochs['face']]
print max_face_Epoch
print max_face_Epoch[:5]

max_scramble_Epoch = [e.max() for e in epochs['scramble']]
print max_scramble_Epoch
print max_scramble_Epoch[:5]


#as far as I can tell, for Epoch objects no built-in code to:
    #detrend
    #baseline
    #filter
    #--> has functions for Raw objects but not Epoch objects


#mne.epochs.detrend --> imported by epochs.py from mne.filter.py
    #help(mne.filter.detrend)
    #--> input data needs to be an n-d array, not an Epoch object

#mne.epochs.rescale --> imported by epochs.py from mne.baseline.py
    #help(mne.epochs.rescale)
mne.epochs.rescale(epochs,time=,baseline=,mode=)





mne.epochs.subtract_evoked
mne.epochs.crop   #crops a time interval from epochs object



###############################################################################
# Plot individual trials of Epochs

epochs_face = epochs['face']
epochs_scramble = epochs['scramble']

mne.viz.plot_epochs(epochs, epoch_idx=None, picks=[1])

mne.viz.plot_epochs(epochs_face, epoch_idx=None, picks=[1])
mne.viz.plot_epochs(epochs_scramble, epoch_idx=None, picks=[1])


###############################################################################
# Calculate Evoked & plot

#average ALL trials
evoked_All = epochs.average()
#average only FACE trials
evoked_face = epochs['face'].average()
#average only the last 5 trials
evoked_last5trials = epochs[-5:].average()

#epochs.average() --> uses np.mean (no root mean square in epochs.py)
#epochs.standard_error() --> uses np.std, then converts to standard error

evoked_scramble_average = epochs['scramble'].average()
evoked_scramble_sterror = epochs['scramble'].standard_error()
print evoked_scramble_average
evoked_scramble_average.plot()

evoked_face_average = epochs['face'].average()
evoked_face_sterror = epochs['face'].standard_error()
print evoked_face_average
evoked_face_average.plot()


#save Evoked as .fif if desired
evoked_scramble_average.save('train_subject01_evokedScrambleAverage.fif')
evoked_face_average.save('train_subject01_evokedFaceAverage.fif')


#read in Evoked .fif and apply a baseline
saved_evoked = mne.fiff.read_evoked('/home/tobin/Desktop/train_subject01_evokedScrambleAverage.fif', setno=None, baseline=(-100, 0), kind='average', proj=False)
#--> can apply a baseline correction!


###############################################################################
# Calculate Evoked contrast & plot

face_minus_scrambled_contrast = evoked_face_average - evoked_scramble_average
print face_minus_scrambled_contrast
face_minus_scrambled_contrast.plot()


###############################################################################
# Show event related fields images

import matplotlib.pyplot as plt

# and order with spectral reordering
# If you don't have scikit-learn installed set order_func to None
from sklearn.cluster.spectral import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel


def order_func(times, data):
    this_data = data[:, (times > 0.0) & (times < 0.350)]
    this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
    return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.),
                      n_components=1, random_state=0).ravel())

epochs_face=epochs['face']
epochs_scramble=epochs['scramble']



#if 'picks=None' will plot all channels listed as 'good'

#plot 5 channels, epochs in no order
plt.close('all')
mne.viz.plot_image_epochs(epochs_face, [11, 12, 13, 14, 15], sigma=0.5, vmin=-100,
                          vmax=250, colorbar=True, order=None, show=True)

#plot 1 channel, epochs in no order
channel=1
plt.close('all')
mne.viz.plot_image_epochs(epochs_face, picks=channel, sigma=0.5, vmin=-100,
                          vmax=250, colorbar=True, order=None, show=True)


#plot two channels, order the epochs using the function defined above
good_pick = 97
bad_pick = 98

plt.close('all')
mne.viz.plot_image_epochs(epochs, [good_pick, bad_pick], sigma=0.5, vmin=-100,
                          vmax=250, colorbar=True, order=order_func, show=True)



###############################################################################
# Export

#export Epochs to Pandas
epochs_df = epochs.as_data_frame()

#export Epochs to Nitime
import nitime
epochs_ts = epochs.to_nitime()




