"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: Generates an SVM for the first subject, with 20 sessions, from the NSD dataset.
"""
#%% Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#%% Load in the combined data for one subject, 20 sessions, and compile to single numpy array by vertically stacking sessions
# Todo: check this code to ensure indexing matches as expected
file_path = os.getcwd()+'/data_processing/sub1_ses1_to_20/'
sub1_ses1_files = os.listdir(file_path)
sub1_all_sessions = np.zeros((1491, 114264))
sum_row = 0
idx = 0
for count in range(len(sub1_ses1_files)):
    session = np.load(file_path + f"subj1_ses{count + 1}.npy")
    print(np.shape(session))
    sub1_all_sessions[idx: idx+np.shape(session)[0], :] = session
    idx = np.shape(session)[0]

# save all sessions data
np.save('sub1_allses.npy', sub1_all_sessions)

#%% Reload data
X = np.load(os.getcwd() + '/data_processing/sub1_allses.npy')

# generate labels for all sessions from saved full subjects, full sessions dataset 'sample_input_outpu.csv'
targets_path = os.getcwd() + '/annotation_data/sample_input_output.csv'
y = list(pd.read_csv(targets_path)['final_socnonsoc'])[0:1491] # 1491 is used because it is the number of trials in the first 20 sessions for subject 1

#%% Perform SVM
# Split the X and y data
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

# Pre-process with StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Make SVC classfier with linear kernel and balanced class weights
# instantiate classifier with default hyperparameters
svc=SVC(kernel='linear', class_weight='balanced')

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% Determine the indices of the maximum coefficients to use for visualization
coefficients = list(svc.coef_[0])
print(f'The number of nonzero values is: {np.count_nonzero(coefficients)}')
max_coef_idx = coefficients.index(max(coefficients)) # maximum coefficient
coefficients.remove(max(coefficients))
second_max_coef_idx = coefficients.index(max(coefficients)) # next largest coefficient

#%% Check for overfitting -- print the scores on training and test set
print('Training set score: {:.4f}'.format(svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(svc.score(X_test, y_pred)))

# Check predictions in test set
print('The predicted labels are:')
print(y_pred)

# Show the null accuracy
null_accuracy = (np.sum(y_pred)/(len(y_pred)))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#%% Plot with indices for x, y axes specified by max coefficients
# Plot training set
labels = y_train
colordict = {0:'red', 1:'blue'}
piclabel = {0:'Non-Social', 1:'Social'}
markers = {0:'o', 1:'o'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))

for m in np.unique(labels):
    ix1 = np.where(labels == m)[0]
    plt.scatter(x=list(np.array(X_train.iloc[ix1, max_coef_idx]).transpose()),
                    y=list(np.array(X_train.iloc[ix1, second_max_coef_idx]).transpose()), c = colordict[m],
                    label = piclabel[m], s = 40, marker = markers[m], alpha = alphas[m])
plt.title('Train')
plt.show()

# Plot test set
labels = y_train
colordict = {0:'red', 1:'blue'}
piclabel = {0:'Non-Social', 1:'Social'}
markers = {0:'o', 1:'o'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))

for m in np.unique(labels):
    ix1 = np.where(labels == m)[0]
    plt.scatter(x=list(np.array(X_test.iloc[ix1, max_coef_idx]).transpose()),
                    y=list(np.array(X_test.iloc[ix1, second_max_coef_idx]).transpose()), c = colordict[m],
                    label = piclabel[m], s = 40, marker = markers[m], alpha = alphas[m])

plt.title('Test')
plt.show()
