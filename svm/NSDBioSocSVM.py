"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: This file contains the model class for constructing an SVM for the NSD biosoc data.
The current version uses only subject 1, session 1, data.

Possible tutorial: https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook

"""
#%% Import packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Todo: make variable names explictly identifiable
# Todo: clean up script

#%% Generate input and output
X = np.load(os.getcwd()+'/data_processing/flattened_dataset/flattened_downsampled_session1_sub1.npy')
y = list(np.load(os.getcwd() + '/data_processing/flattened_dataset/labels_session1_sub1.npy'))

#%% Split the X and y data
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

#%% Pre-process with StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#%% Make SVC classfier with linear kernel and balanced class weights
# instantiate classifier with default hyperparameters
svc=SVC(kernel='linear', class_weight='balanced')

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% Determine indices of the maximum coefficients to use for visualization
coefficients = list(svc.coef_[0])
print(f'The number of nonzero values is: {np.count_nonzero(coefficients)}')
max_coef_idx = coefficients.index(max(coefficients))
coefficients.remove(max(coefficients))
second_max_coef_idx = coefficients.index(max(coefficients))

#%% Check for overfitting -- print the scores on training and test set
print('Training set score: {:.4f}'.format(svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(svc.score(X_test, y_pred)))

# Check predictions in test set
print('The predicted labels are:')
print(y_pred)

# Show the null accuracy
null_accuracy = (np.sum(y_pred)/(len(y_pred)))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# Todo: check that confusion matrix performs as expected
#%% Plot confusion matrix - this code may need to be further adapted
# clf = SVC(random_state=0)
# clf.fit(X_train, y_train)
# plot_confusion_matrix(clf, X_test, y_test)
# plt.show()

# Todo: ensure that Grid search performs as expected
#%% Grid search -- this code may need to be further adapted
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
#
# # Set the parameters by cross-validation
# tuned_parameters = [
#     {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
#     {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
# ]
#
# scores = ["precision", "recall"]
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_["mean_test_score"]
#     stds = clf.cv_results_["std_test_score"]
#     for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
#         print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()


#%% Plot training set with indices for x, y axes specified by max coefficients
import seaborn as sns
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
plt.savefig('figures/tran_sub1_ses1.png')
plt.show()

#%% Plot test set

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
plt.savefig('figures/test_sub1_ses1.png')
plt.show()

# Todo: complete code to plot hyperplanes
#%% This code is taken from an online source and must be adapted.
# # Constructing a hyperplane using a formula.
# w = svc.coef_[0]           # w consists of 2 elements
# b = svc.intercept_[0]      # b consists of 1 element
# x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
# y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# # Plotting a red hyperplane
# plt.plot(x_points, y_points, c='r');
# # Encircle support vectors
# plt.scatter(svc.support_vectors_[:, 0],
#             svc.support_vectors_[:, 1],
#             s=50,
#             facecolors='none',
#             edgecolors='k',
#             alpha=.5);
# # Step 2 (unit-vector):
# w_hat = svc.coef_[0] / (np.sqrt(np.sum(svc.coef_[0] ** 2)))
# # Step 3 (margin):
# margin = 1 / np.sqrt(np.sum(svc.coef_[0] ** 2))
# # Step 4 (calculate points of the margin lines):
# decision_boundary_points = np.array(list(zip(x_points, y_points)))
# points_of_line_above = decision_boundary_points + w_hat * margin
# points_of_line_below = decision_boundary_points - w_hat * margin
# # Plot margin lines
# # Blue margin line above
# plt.plot(points_of_line_above[:, 0],
#          points_of_line_above[:, 1],
#          'b--',
#          linewidth=2)
# # Green margin line below
# plt.plot(points_of_line_below[:, 0],
#          points_of_line_below[:, 1],
#          'g--',
#          linewidth=2)