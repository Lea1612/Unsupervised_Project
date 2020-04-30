#UNSUPERVISED - Pr Yoram Louzoun

#Lea Setruk 345226179
#Aviva Shneor Simchon 317766731

import numpy as np
import pandas as pd
import sklearn
import scipy 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import classification_report,accuracy_score,silhouette_score,fowlkes_mallows_score, v_measure_score
from sklearn import metrics
from time import time
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.cluster import  DBSCAN
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.utils import resample
import sys

#rcParams['figure.figsize'] = 14, 8
#RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


#data
#data = pd.read_csv(r'C:\Users\utilisateur\Documents\Master 1\Unsupervised\Unsupervised\creditcard.csv', na_values="?",
                # low_memory=False)

#data.info()



file = sys.argv[1]

data = pd.read_csv(file, na_values="?", low_memory=False)

#nb of observations
n = data.shape[0]
#nb of variables
p = data.shape[1]

print ( n,p)



# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()
data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!
print(data.shape)
data.head()




#reduce  the data with no transform
#take for few iteration sample of 5000 normal and all the fraud
data_for_dbscan= data.copy()

fraud1 = data_for_dbscan[data_for_dbscan['Class'] ==1]
normal1 = data_for_dbscan[data_for_dbscan['Class'] ==0]

data_normal_sampled = resample(normal1, n_samples=5000,random_state=None)
data_reduce = pd.concat([fraud1, data_normal_sampled], axis=0).sample(frac=1).reset_index(drop=True)
print(data_for_dbscan.Class.value_counts())



print(fraud1.shape,data_normal_sampled.shape)
outlier_fraction = len(fraud1)/float(len(data_normal_sampled))
print(outlier_fraction)



data_reduce.head()



#Sample of the data
X = data_reduce[['V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18','scaled_time']]
y = data_reduce['Class']
print(X.shape)
print(y.shape)


X_all = data[['V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18','scaled_time']]
y_all = data['Class']
print(X_all.shape)
print(y_all.shape)


#MDS

#t0 = time()
mds = manifold.MDS(2, max_iter=30, n_init=1)
trans_data = mds.fit_transform(X)
#t1 = time()
#print("MDS: %.2g sec" % (t1 - t0))


tempdf =pd.DataFrame(trans_data)
tempdf.head()
mds_df_with_true_label= tempdf.copy()
mds_df_with_true_label["Class"]=y
mds_df_with_true_label = mds_df_with_true_label.rename(columns={0: "Dim_1", 1: "Dim_2"})
mds_df_with_true_label.head()



plt.figure(figsize=(8,4))
sns.scatterplot(x="Dim_1",y="Dim_2",hue="Class",data= mds_df_with_true_label,palette =sns.color_palette("hls",2))
ax = plt.gca()
ax.set_title("Data after MDS dimension reduce")



#plotting the data
# perform PCA for plotting
pca = PCA(n_components=2)
data_pca = pca.fit_transform(X_all)

df_data_pca = pd.DataFrame(data_pca)
df_data_pca["y"]= y_all



plt.figure(figsize=(8,4))
sns.scatterplot(x=0,y=1,hue="y",data= df_data_pca,palette =sns.color_palette("hls",2),alpha= 0.6)
ax = plt.gca()
ax.set_title("Data after PCA dimension reduce")



# clusterer = hdbscan.HDBSCAN(min_cluster_size=100).fit(X)
# color_palette = sns.color_palette('deep', 8)
# cluster_colors = [color_palette[x] if x >= 0
# else (0.5, 0.5, 0.5)
# for x in clusterer.labels_]
# cluster_member_colors = [sns.desaturate(x, p) for x, p in
# zip(cluster_colors, clusterer.probabilities_)]
# plt.scatter(*X.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)




# clusterer.labels_.max()



# data=tempdf.copy
# clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(data)
# color_palette = sns.color_palette('Paired', 12)
# cluster_colors = [color_palette[x] if x >= 0
# else (0.5, 0.5, 0.5)
# for x in clusterer.labels_]
# cluster_member_colors = [sns.desaturate(x, p) for x, p in
# zip(cluster_colors, clusterer.probabilities_)]
# plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)



# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
#                                       edge_alpha=0.6,
#                                       node_size=80,
#                                       edge_linewidth=2)


# # DBSCAN


neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(8,4))
plt.plot(distances)

#distances.remove('42.76952509128829')
#distances = distances[distances != 42.640396986629]
#print(*distances, sep = "\n") 

print('average')
print(np.mean(distances, axis=None, dtype=None, out=None))
print(X.shape)
minsample = np.log(len(X))
print(minsample)



df = X.copy()
for col in df.columns:
    avg = df[col].mean()
    sd = df[col].std()
    df[col] = df[col].apply(lambda x: (x-avg)/sd)
#df.head


minsample = np.log(5465)
print(minsample)



#db = DBSCAN(eps=3.3, min_samples=9)
#db = DBSCAN(eps=2.6, min_samples=9)#Silhouette Coefficient: 0.614
#db = DBSCAN(eps=2.2, min_samples=9)#Silhouette Coefficient: 0.598
#db = DBSCAN(eps=1.9, min_samples=9)#better Silhouette Coefficient: 0.520
#db = DBSCAN(eps=1.4, min_samples=9)#נקאאקר
db = DBSCAN(eps=5.8 ,min_samples=8.8)
db_label = db.fit_predict(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################


#mds = manifold.MDS(2, max_iter=30, n_init=1)
#trans_data = mds.fit_transform(X)

temp_df = pd.DataFrame(trans_data)
temp_df.head()



temp_df['pred_labels'] = db_label
temp_df["pred_labels"].value_counts()
df_dbscan = pd.DataFrame(temp_df)
df_dbscan["Class"]=mds_df_with_true_label["Class"]
df_dbscan.head()
pd.crosstab(df_dbscan['pred_labels'],df_dbscan['Class'])



df_dbscan = pd.DataFrame(temp_df)
df_dbscan["Class"]=mds_df_with_true_label["Class"]
df_dbscan.head()
pd.crosstab(df_dbscan['pred_labels'],df_dbscan['Class'])



#plt.figure(figsize=(8,6))
#sns.scatterplot(x=0, y=1, hue="pred_labels", data=temp_df, palette=sns.color_palette("hls",7), alpha=0.5, legend="full")


#plot after MDS
fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (14,10))
#sns.scatterplot(x=0, y=1, hue="pred_labels", data=df_dbscan, palette=sns.color_palette("hls",4),  legend="full",ax=ax1)
sns.scatterplot(x=0, y=1, hue="pred_labels", data=df_dbscan, palette=sns.color_palette("hls",5),  legend="full",ax=ax1)
ax1.set_title("DBSCAN ")

plt.sca(ax2)
sns.scatterplot(x=0, y=1, hue="Class", data=df_dbscan, palette=sns.color_palette("hls",2),  legend="full",ax=ax2)
ax2.set_title("Data after MDS ")

sns.scatterplot(x=0, y=1, hue="Class", data=df_dbscan, palette=sns.color_palette("hls",2),  legend="full")

centers = db.core_sample_indices_
#print(centers)
X=df_dbscan.loc[:,[0,1,'pred_labels']]
X.plot.scatter(x = 0, y = 1, c=db_label, s=50, cmap='viridis')
plt.scatter(centers[0], centers[1],c='black', s=200, alpha=0.3)




