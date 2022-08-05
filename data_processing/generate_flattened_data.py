"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: scratch file to generate DataFrame containing data
"""
#%% Import packages
import s3fs
import nibabel as nib
import pandas as pd
import os
import numpy as np

#%% Access NSD data from Amazon S3 database
fs = s3fs.S3FileSystem(anon=True)

trials_info = pd.read_csv(os.getcwd() + '/annotation_data/shared_trial_info.csv')
trials_sub1 = trials_info[((trials_info.SUBJECT == 1))]
trials_sub1.head()
len(trials_sub1)

#%% Generate flattened dataset by pulling directly from S3
# 1.8 mm space functional data
# subj 1, session 1
# trials = trials_sub1['trial_in_session'].tolist()
sessions = [1]  # Test with only first session
# sessions = trials_sub1.SESSION.unique() #to loop over all sessions

ll = fs.ls(f'natural-scenes-dataset/nsddata_betas/ppdata/subj01/MNI/betas_fithrf/betas_session0{session}.nii.gz')
fs.get(ll[0], "tmp.nii.gz")
img = nib.load("tmp.nii.gz")
trials_sub_ses = trials_sub1[(trials_sub1.SESSION == session)]
print('Begin trial processing.')
for count, t in enumerate(trials_sub_ses['trial_in_session'].tolist()):
    beta_trial = img.slicer[:, :, :, t].get_fdata() / 300
    flat_beta_trial = beta_trial.flatten()
    if count == 0:
        full_betas = flat_beta_trial
        print(np.shape(full_betas))
    else:
        full_betas = np.vstack((full_betas, flat_beta_trial))
        print(np.shape(full_betas))
        #np.save(os.getcwd() + f'/data_processing/flattened_dataset/{count}_beta', beta_trial)

#%% Load saved numpy arrays to generate combined array for Session 1 Subject 1
files = os.listdir(os.getcwd() + '/data/session1_subject1_ betas/')
for i in range(0,62):
    flat_beta_trial = np.load(os.getcwd() + '/data/session1_subject1_ betas/' + f"{i}_beta.npy").flatten()
    if i == 0:
        full_betas = flat_beta_trial
        print(np.shape(full_betas))
    else:
        full_betas = np.vstack((full_betas, flat_beta_trial))
        print(np.shape(full_betas))

#%% Save flattened data -- probably slow
np.savetxt(os.getcwd()+'/data_processing/flattened_dataset/flattened_session1_sub1.csv', full_betas, delimiter=",")

#%% Try reloading -- probably slow
reloaded_betas = np.genfromtxt(os.getcwd() + '/data_processing/flattened_dataset/flattened_session1_sub1.csv')

#%% Save first 62 labels
csv_file = os.getcwd() + '/annotation_data/sample_input_output.csv'
labels = list(pd.read_csv(csv_file).iloc[0:62, 2])
label_array = np.array(labels)
np.save(os.getcwd()+'/data_processing/flattened_dataset/labels_session1_sub1.npy', label_array)

#%% Save numpy.ndarray as numpy file
np.save(os.getcwd()+'/data_processing/flattened_dataset/flattened_session1_sub1.npy', full_betas)

#%% Reduce to every other column in
full_betas_copy_filtered = full_betas.copy()[:, 1::2]

#%% Save filtered array
np.save(os.getcwd()+'/data_processing/flattened_dataset/flattened_downsampled_session1_sub1.npy', full_betas_copy_filtered)
