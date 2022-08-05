"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: script to run PCA on a single subject and single session
"""
#%% Import packages
import os
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% Load the data
X = np.load(os.getcwd()+'/data_processing/flattened_dataset/flattened_downsampled_session1_sub1.npy')
y = list(np.load(os.getcwd() + '/data_processing/flattened_dataset/labels_session1_sub1.npy'))

#%% PCA
#Scale the data
scaler = StandardScaler()
feature_scaled = scaler.fit_transform(X)

#Apply PCA
pca = PCA(n_components=3)
pca.fit(feature_scaled)
feature_scaled_pca = pca.transform(feature_scaled)
np.shape(feature_scaled_pca)

#%% Two dimensional plotting
feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print ("Variance Ratio of the 3 Principal Components Ananlysis: ", feat_var_rat)

target_list = y

feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]

labels = target_list
colordict = {0:'red', 1:'blue'}
piclabel = {0:'Non-Social', 1:'Social'}
markers = {0:'o', 1:'o'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
for l in np.unique(labels):
    ix = np.where(labels==l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l],
               label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.subplot(1,2,2)
for l1 in np.unique(labels):
    ix1 = np.where(labels==l1)
    plt.scatter(feature_scaled_pca_X0[ix1], feature_scaled_pca_X2[ix1], c=colordict[l1],
               label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Third Principal Component", fontsize=15)

plt.subplot(2,2,2)
for l1 in np.unique(labels):
    ix1 = np.where(labels==l1)
    plt.scatter(feature_scaled_pca_X1[ix1], feature_scaled_pca_X2[ix1], c=colordict[l1],
               label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("Second Principal Component", fontsize=15)
plt.ylabel("Third Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.show()

#%% 3D plot
labels = target_list
colordict = {0:'red', 1:'blue'}
piclabel = {0:'Non-Social', 1:'Social'}
markers = {0:'o', 1:'o'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(projection='3d')

for m in np.unique(labels):
    ix1 = np.where(labels == m)
    xs = feature_scaled_pca_X0[ix1]
    ys = feature_scaled_pca_X1[ix1]
    zs = feature_scaled_pca_X2[ix1]
    ax.scatter(xs, ys, zs, label=piclabel[m], marker=markers[m])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(['Social', 'Non-Social'])
plt.show()

