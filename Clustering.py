#UNSUPERVISED - Pr Yoram Louzoun

#Lea Setruk 345226179
#Aviva Shneor Simchon 317766731

import numpy as np
import pandas as pd
import sklearn
#import scipy 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import classification_report,accuracy_score,silhouette_score,fowlkes_mallows_score, v_measure_score
from pylab import rcParams
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import power_transform
from sklearn.utils import resample
from sklearn import metrics
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.utils import shuffle
from warnings import simplefilter
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys

rcParams['figure.figsize'] = 14, 8
#RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

input_file = sys.argv[1]

#data_init = pd.read_csv(r'C:\Users\user\Projet 2020 Bar Ilan\creditcard.csv', na_values="?",low_memory=False)
#data_init = pd.read_csv(r'C:\Users\utilisateur\Documents\Master 1\Unsupervised\Unsupervised\creditcard.csv', na_values="?",low_memory=False)
data_init = pd.read_csv(input_file, na_values="?",low_memory=False)



#nb of observations
#n = data.shape[0]
#nb of variables
#p = data.shape[1]

#print ( n,p)

#data.head(4)

#Empty in the data ?
#data_init.isnull().values.any()


#Time and Amount scalling

# RobustScaler is less prone to outliers.
data = data_init.copy()

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


#PCA For ALL Data and ALL features
#במידה והמטרה היא לפרתח מודל פרדיקטיבי יש להיזהר כי לא חולק לTRAIN ןTEST
df_for_pca = data.copy()

del df_for_pca['Class']

pca_model_2Dim = PCA(n_components=2)
pca_data_2Dim = pca_model_2Dim.fit_transform(df_for_pca)

df_pca = pd.DataFrame(pca_data_2Dim)

#print(np.array2string(pca_model.components_,precision=3, suppress_small=True))

df_pca['Class'] = data['Class']
sns.scatterplot(x=0,y=1,hue="Class",data=df_pca,palette =sns.color_palette("hls",2))


#Graph of variables with statistical description
for c in data.columns[0:30]:
    norm = data[c].mean() + 3*data[c].std()
    data2 = data[data[c].abs() < norm]
    col= data[data[c].abs() < norm][c]
    col=np.array(col)
    col_mean= np.mean(col)
    col_median= np.median(col)
    col_mode= stats.mode(col)[0][0]
    col_std= np.std(col)
    col_var= np.var(col)
    col_range= col.max()-col.min()
    fig=sns.FacetGrid(data2,hue="Class",height=5,aspect=2,palette=["blue", "green"])
    fig.map(sns.distplot,c)
    fig.add_legend(labels=['Normal','Fraud'])
    #plt.yscale('log')
    plt.axvline(col_mean,color='red',label='mean')
    plt.axvline(col_median,color='yellow',label='median')
    plt.axvline(col_mode,color='black',label='mode')
    plt.legend()
    plt.show


#show the histogram better.witno outlier
#sort the value and remove outliers 

df = pd.DataFrame()

for c in data.columns:
    #plt.figure(i)
    if c[0]!= 'V':
        continue
    x = data[c]
    x= x.sort_values()
    x= x[20000:-20000]
    df[c]=x


df.hist(bins = 50, figsize = (20,20))
plt.show()


#now we transform the data to Gaussian and delete outliers to see better evry features


dfT = pd.DataFrame()
for c in data.columns:
    #plt.figure(i)
    if c[0]!= 'V':
        continue
    x = data[c]
    x= x.sort_values()
    x= x[20000:-20000]
    #x=sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    X=power_transform(x[:,np.newaxis], method='yeo-johnson', standardize=True, copy=True)

    #print(X.shape)
    dfT[c]=X.squeeze()


dfT.hist(bins = 50, figsize = (20,20))
plt.show()


#Transform the data for work - no delete outlier
#we use method 'yeo-johnson' because we have negative values
dataT = pd.DataFrame()
for c in data.columns:
    #plt.figure(i)
    if c[0]!= 'V':
        continue
    x = data[c]
    X=power_transform(x[:,np.newaxis], method='yeo-johnson', standardize=True, copy=True)
    dataT[c]=X.squeeze()


#BoxPlot
i=0
for c in data.columns:
    plt.figure(i)
    norm = data[c].mean() + 3*data[c].std()
    data[data[c].abs() < norm].boxplot([c], 'Class')
    i+=1



colors = ['#B3F9C5', '#f9c5b3']
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS )
plt.yscale('log')
plt.xlabel("Class")
plt.ylabel("Frequency")


# Get the Fraud and the normal dataset

fraud = data[data['Class']==1]
normal = data[data['Class']==0]


# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.
fraud_init = data_init[data_init['Class']==1]
normal_init = data_init[data_init['Class']==0]
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud_init.Time/3600, fraud_init.Class)
ax1.set_title('Fraud')

ax2.scatter(normal_init.Time/3600, normal_init.Class)
ax2.set_title('Normal')
plt.xlabel('Time (in hour)')
plt.ylabel('Amount')
plt.show()



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud_init.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal_init.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();



# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud_init.Time/3600, fraud_init.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal_init.Time/3600, normal_init.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in hour)')
#plt.ylabel('Amount')covMatrix = np.cov(Data,bias=True)
#print (covMatrix)
plt.show()

fraud.describe()

normal.describe()


#get correlations of each features in dataset
corrmat = data.corr()

print(corrmat)

  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

plt.figure(figsize=(20,50))
plt.subplot(11,3,1)
sns.kdeplot(normal['scaled_amount'], shade= True)
sns.kdeplot(fraud['scaled_amount'], shade= True)
plt.tight_layout()
plt.yscale('log')
plt.show()

plt.figure(figsize=(20,50))
plt.subplot(11,3,2)
sns.kdeplot(normal['scaled_time'], shade= True)
sns.kdeplot(fraud['scaled_time'], shade= True)
plt.tight_layout()
plt.show()
   


# we can see that some feature are distributed the same for fraud and normal so we won't take them for the analyze

# we choose features: 'V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18'


#הדטה ממש גדול ולכן ניקח חלק בשביל העבודה
## Take some sample of the data
data_transform=pd.DataFrame()
data_transform = dataT.copy()

data_transform['scaled_time'] = data['scaled_time']
data_transform['scaled_amount'] = data['scaled_amount']
data_transform['Class'] = data['Class']
#data_transform.head()



data_transform_reduce=pd.DataFrame()
#reduce  the data
#take for few iteration sample of 5000 normal and all the fraud
fraud1 = data_transform[data_transform['Class'] ==1]
normal1 = data_transform[data_transform['Class'] ==0]

data_normal_sampled = resample(normal1, n_samples=5000, random_state=None)
#data_transform[data_transform['Class'] ==0][:round(len(data_transform)/2)]
data_transform_reduce = pd.concat([fraud1, data_normal_sampled], axis=0).sample(frac=1).reset_index(drop=True)
print(data_transform_reduce.Class.value_counts())


print(fraud1.shape,data_normal_sampled.shape)
outlier_fraction = len(fraud1)/float(len(data_normal_sampled))
print(outlier_fraction)

#features = data_transform_reduce



#Sample of the data
X = data_transform_reduce[['V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18','scaled_time']]
y = data_transform_reduce['Class']
#print(X.shape)
#print(y.shape)


#X.head()



#All the data
X_all_data = data_transform[['V1','V2','V3','V4','V5','V7','V9','V10','V11','V12','V14','V16','V17','V18','scaled_time']]
y_all_data = data_transform['Class']

#print(X_all_data.shape)
#print(y_all_data.shape)


# general functions


def plot_3d_scatter(data,labels):
    #plt.subplot(1,2,1)
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.scatter(data[:,1], data[:,0], data[:,2],c=labels)
    #plt.subplot(1,2,2)
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.scatter(data[:,0], data[:,1], data[:,2],c=labels)

    plt.show()



def print_silouhette_score(data,model_label):
    print("silhouette score : %0.3f" %  silhouette_score(data,model_label))



def compute_mutual_info_p_value(true_label,model_label):
    simplefilter(action='ignore', category=FutureWarning)
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    p = 1000
    vec = []
    real_score = normalized_mutual_info_score(true_label,model_label)
    #print(real_score, "real score")
    for i in range(p):
        y_shuffle = shuffle(true_label).reset_index()
        y_shuffle = y_shuffle.drop('index', axis=1)
        #print(y_shuffle)
        y_shuffle = y_shuffle.squeeze()
        #dummy_score = normalized_mutual_info_score(y_shuffle, Kmean_cluster)
        dummy_score = normalized_mutual_info_score(y_shuffle, model_label)
        #if dummy_score > homo_score:
           # print(high)
        vec.append(dummy_score)
  
    p_value_s = np.array(vec)

   
    print('p value for normalized mutual info- ',len(p_value_s[p_value_s > real_score]) / float(1000))
    sns.distplot(p_value_s, axlabel="normalized mutual info")



def print_clustering_score(true_label,model_cluster):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_label,model_cluster))
    print("Completeness: %0.3f" % metrics.completeness_score(true_label,model_cluster))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_label,model_cluster))
    print("normalized mutual info: %0.3f" % normalized_mutual_info_score(true_label,model_cluster))
    compute_mutual_info_p_value(true_label, model_cluster)


# PCA



pca_model_3 = PCA(n_components=3)
pca_3_data_sample = pca_model_3.fit_transform(X)

pca_model_2 = PCA(n_components=2)
pca_2_data_sample = pca_model_2.fit_transform(X)

pca_2_all_data = pca_model_2.fit_transform(X_all_data)


# Clustering

# K-mean



#def find_best_num_clusterKMean(data):
elbow_scores = {}
fowlkes_scores = {}
v_measure_scores = {}
silhouette_scores = {} 

for k in range(2,9):
    #print(k)
    kmeans_model = KMeans(n_clusters=k,n_jobs=-1)
    kmeans_model.fit(X)
    Kmean_Pred = kmeans_model.labels_

    elbow_scores.update({k:kmeans_model.inertia_})
    fowlkes_scores_val = fowlkes_mallows_score (y,Kmean_Pred)
    fowlkes_scores.update({k:fowlkes_scores_val})
    v_measure_val = v_measure_score(y,Kmean_Pred)
    v_measure_scores.update({k:v_measure_val})
    silhouette_score_val = silhouette_score(X,kmeans_model.labels_)
    silhouette_scores.update({k:silhouette_score_val})




#Sןilouhette score
plt.figure(figsize=(7,4))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=list(silhouette_scores.keys()),y=list(silhouette_scores.values()),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()



# Plot sse against k
plt.figure(figsize=(8, 4))

plt.title('The Elbow Method using Inertia') 
plt.plot(list(elbow_scores.keys()),list(elbow_scores.values()), '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


#fowlkes

plt.bar(list(fowlkes_scores.keys()),list(fowlkes_scores.values()),align='center')
plt.xlabel("Number of clusters")
plt.ylabel("fowlkes_scores")
plt.title('fowlkes score K-means')
plt.plot()



#vmeasure
#plt.bar(list(v_measure_scores.keys()),list(v_measure_scores.values()),align='center', alpha=0.5)
plt.scatter(list(v_measure_scores.keys()),list(v_measure_scores.values()),s=150,edgecolor='k')
plt.xlabel("Number of clusters")
plt.ylabel("v_measure_scores")
plt.title('v_measure_scores K-means')
plt.plot()


# # KMEAN MODEL


# we see that the scores give us 3 clusters

# run kmeans algorithm
kmeans_model = KMeans(n_clusters=3,n_jobs=-1)
kmeans_model.fit(X)

cluster = kmeans_model.labels_
centroids = kmeans_model.cluster_centers_
Kmean_cluster = kmeans_model.predict(X)
# # perform PCA for plotting
# pca_model_3 = PCA(n_components=3)
# kmeans_pca_data = pca_model_3.fit_transform(X)

# plot 3D scatter
#plot_3d_scatter(kmeans_pca_data,Kmean_cluster)

# perform MDS for plotting
Kmean_mds_data = manifold.MDS(2, max_iter=10, n_init=1).fit_transform(X)
df_mds = pd.DataFrame(Kmean_mds_data)
df_mds["Cluster"]  = Kmean_cluster
df_mds["Class"] = y

#plot after MDS
fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))
sns.scatterplot(x=0,y=1,hue="Cluster",data=df_mds ,palette =sns.color_palette("hls",3),ax=ax1)
ax1.set_title("K- mean Clustering. visualize on 2 dim after MDS")

plt.sca(ax2)

sns.scatterplot(x=0,y=1,hue="Class",data=df_mds,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("Data after MDS ")


#KMean analyze clusters
Kmean_df = pd.DataFrame()
Kmean_df = X.copy()

Kmean_df['Cluster'] = pd.Series(Kmean_cluster)
Kmean_df['Class']  = y

Kmean_df.head()
pd.crosstab(Kmean_df['Cluster'],Kmean_df['Class'])


print_clustering_score(y, Kmean_cluster)
print_silouhette_score(X,Kmean_cluster)
#compute_mutual_info_p_value(y, Kmean_cluster)


# K-means after PCA 



#K-MEAN AFTAR PCA
# perform PCA 2 dimensions for plotting
#model_pca_2 = PCA(n_components=2)
#data_pca = model_pca_2.fit_transform(X)
#pca_2_data_sample
#print(model_pca.components_)
df_pca = pd.DataFrame()
df_pca = pd.DataFrame(pca_2_data_sample)
##########################################
kmeans_model_pca = KMeans(n_clusters=3,n_jobs=-1).fit(df_pca.values)
kmeans_pca_cluster = kmeans_model_pca.predict(df_pca.values)

df_pca["Cluster"]  = kmeans_pca_cluster
df_pca["Class"] = y

pd.crosstab(df_pca['Cluster'],df_pca['Class'])
#################################################
fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))
sns.scatterplot(x=0,y=1,hue="Cluster",data=df_pca ,palette =sns.color_palette("hls",3),ax=ax1)
ax1.set_title("K-mean on data after PCA dimension reduce to 2 dim")

plt.sca(ax2)

sns.scatterplot(x=0,y=1,hue="Class",data=df_pca,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("Data after PCA ")
#df_pca.head()
pd.crosstab(df_pca['Cluster'],df_pca['Class'])





feature = pd.DataFrame()
feature = X.columns

cluster_0 = Kmean_df[Kmean_df['Cluster'] ==0]
cluster_1 = Kmean_df[Kmean_df['Cluster'] ==1]
cluster_2 = Kmean_df[Kmean_df['Cluster'] ==2]

plt.figure(figsize=(20,50))
i=1
for feat in feature:
    plt.subplot(11,3,i)
    sns.kdeplot(cluster_0[feat], shade= True)
    sns.kdeplot(cluster_1[feat], shade= True)
    sns.kdeplot(cluster_2[feat], shade= True)
    i=i+1
    plt.tight_layout()
    #plt.yscale('log')
plt.show()



print_clustering_score(y, kmeans_pca_cluster)
print_silouhette_score(X,kmeans_pca_cluster)




#best score because itharvut
print_clustering_score(y, kmeans_pca_cluster)


# predict K mean model for all data



#X_all_data.head()



df= pd.DataFrame()



# Run Predict to all data with the model

#cluster_all_data = kmeans_model.labels_
Kmean_cluster = kmeans_model.predict(X_all_data)

# perform PCA for plotting
#pca_2_all_data = model_pca_2.fit_transform(X_all_data)
#MDS
#Kmean_mds_data = manifold.MDS(2, max_iter=10, n_init=1).fit_transform(X_all_data)
#df_mds = pd.DataFrame(Kmean_mds_data)
#df_mds["Cluster"]  = Kmean_cluster
#df_mds["Class"] = y_all_data
    
df=pd.DataFrame(pca_2_all_data)

df["Cluster"]  = Kmean_cluster
df["Class"] = y_all_data
#df.groupby(["Cluster"])["Class"].count()

pd.crosstab(df['Cluster'],df['Class'])



fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (10,4))

sns.scatterplot(x=0,y=1,hue="Cluster",data=df ,palette =sns.color_palette("hls",3),ax=ax1)
ax1.set_title("Data after PCA and K- mean For all the data")
plt.sca(ax2)
sns.scatterplot(x=0,y=1,hue="Class",data=df,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("Data after PCA ")



print_clustering_score(y_all_data, Kmean_cluster)
#print_silouhette_score(X_all_data,Kmean_cluster) --take too long time to compute




#y_all_data.shape



#pd.crosstab(Kmean_data_all['Cluster'],Kmean_data_all['Class']).plot(kind='bar')
pd.crosstab(df['Cluster'],df['Class']).plot(kind='bar')
plt.yscale('log')


# GMM



#check how many cluster to choose
silhouette_scores = {}
gm_aic = []
gm_bic= []
gm_bicP= []

for i in range(2,8):
    gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X)
    gaussian_mixture_model_labels = gm.fit_predict(X)
    
    silhouette_score_val = silhouette_score(X,gaussian_mixture_model_labels)
    silhouette_scores.update({k:silhouette_score_val}) 
    
    gm_bic.append(-gm.bic(X))
    gm_bicP.append(gm.bic(X))
    gm_aic.append(gm.aic(X))
    
    print(f"key : {silhouette_scores.keys()}")
    print(f"value : {silhouette_scores.values()}")
    



print(gm_bicP)



plt.figure(figsize=(6,3))
plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,8)],y=np.log(gm_bicP),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
plt.xticks([i for i in range(2,8)],fontsize=14)
plt.yticks(fontsize=12)
plt.show()


plt.figure(figsize=(6,3))
plt.title("The Gaussian Mixture model AIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,8)],y=np.log(gm_aic),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Log of Gaussian mixture AIC score",fontsize=15)
plt.xticks([i for i in range(2,8)],fontsize=14)
plt.yticks(fontsize=12)
plt.show()



df_gmm= pd.DataFrame()

gm_model = GaussianMixture(n_components=2).fit(X)
#fit_predict
gm_model_labels = gm_model.predict(X)

print()
# perform PCA for plotting
#pca_2_data_sample = model_pca_2.fit_transform(X)

df_gmm=pd.DataFrame(pca_2_data_sample)

df_gmm["Cluster"]  = gm_model_labels
df_gmm["Class"] = y
#df_gmm.head(3)
pd.crosstab(df_gmm["Cluster"],df_gmm['Class'])



#df_gmm.groupby(["Cluster"])["Class"].count()




pd.crosstab(df_gmm["Cluster"],df_gmm['Class']).plot(kind='bar')
plt.yscale('log')



fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))

sns.scatterplot(x=0,y=1,hue="Cluster",data=df_gmm ,palette =sns.color_palette("hls",2),ax=ax1)
ax1.set_title("GMM Clustering visualise after PCA dimension reduce to 2 Dim  ")

plt.sca(ax2)

sns.scatterplot(x=0,y=1,hue="Class",data=df_gmm,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("Data after PCA reduce to 2 Dim  ")



print_clustering_score(y,gm_model_labels)
silhouette_score_val = silhouette_score(X,gm_model_labels)
print("silhouette score:", silhouette_score_val)


# GMM for all data



# Run Predict to all data with the GM model
#cluster_all_data = kmeans_model.labels_
GM_cluster = gm_model.predict(X_all_data)

# perform PCA for plotting
#gm_pca_data = model_pca_2.fit_transform(X_all_data)

df_gm_all=pd.DataFrame(pca_2_all_data)

df_gm_all["Cluster"]  = GM_cluster
df_gm_all["Class"] = y_all_data
pd.crosstab(df_gm_all["Cluster"],df_gm_all['Class'])

fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))

sns.scatterplot(x=0,y=1,hue="Cluster",data=df_gm_all ,palette =sns.color_palette("hls",2),ax=ax1)
ax1.set_title("GMM model for all the data.visualize in 2 dim (PCA) ")
plt.sca(ax2)
sns.scatterplot(x=0,y=1,hue="Class",data=df_gm_all,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("ALL Data after PCA dim reduce  ")




#print(gm_model.covariances_)




print_clustering_score(y_all_data,GM_cluster)
#print_silouhette_score(X_all_data,GM_cluster) --took long time


# # Agglomerative_model




elbow_scores = {}
silhouette_scores = {}

for k in range(2,6):
    #print(k)
    knn = kneighbors_graph(X,10,n_jobs=-1)
    Agglomerative_model = AgglomerativeClustering(n_clusters=k,connectivity=knn)
    Agglomerative_model.fit(X.values)
    silhouette_score_val = silhouette_score(X,Agglomerative_model.labels_)
    silhouette_scores.update({k:silhouette_score_val})
    



plt.bar(list(silhouette_scores.keys()),list(silhouette_scores.values()),align='center', alpha=0.5)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title('Silhouette score KNN')
plt.plot()



knn = kneighbors_graph(X,10,n_jobs=-1)
Agglomerative_model = AgglomerativeClustering(n_clusters=2,connectivity=knn)
Agglomerative_model.fit(X)
agg_predict = Agglomerative_model.labels_

# perform PCA for plotting
agglomerative_pca = PCA(n_components=3)
agglomerative_pca_data = agglomerative_pca.fit_transform(X)
# plot 3D scatter
plot_3d_scatter(agglomerative_pca_data,Agglomerative_model.labels_)



df_agg = pd.DataFrame(pca_2_data_sample)

df_agg['Cluster'] = pd.Series(agg_predict)
df_agg['Class']  = y
#agg_data.head()
pd.crosstab(df_agg["Cluster"],df_agg['Class'])


fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))
sns.scatterplot(x=0,y=1,hue="Cluster",data=df_agg ,palette =sns.color_palette("hls",2),ax=ax1)
ax1.set_title("Agglomerative model .visualize in 2 dim (PCA) ")
plt.sca(ax2)
sns.scatterplot(x=0,y=1,hue="Class",data=df_agg,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("ALL Data after PCA dim reduce  ")



pd.crosstab(df_agg["Cluster"],df_agg['Class'])




print_clustering_score(y,agg_predict)
print_silouhette_score(X,agg_predict) 


# Run AGG To All data



# Run Predict to all data with the model
knn = kneighbors_graph(X_all_data,10,n_jobs=-1)
Agglomerative_model = AgglomerativeClustering(n_clusters=2,connectivity=knn)
Agglomerative_model.fit(X_all_data)
#agg_predict = Agglomerative_model.labels_
agg_cluster = Agglomerative_model.fit_predict(X_all_data)

df_agg_all_data =pd.DataFrame(pca_2_all_data)
df_agg_all_data["Cluster"]  = agg_cluster
df_agg_all_data["Class"] = y_all_data

pd.crosstab(df_agg_all_data["Cluster"],df_agg_all_data['Class'])




fig, (ax1,ax2)  = plt.subplots(1,2,figsize = (12,4))
sns.scatterplot(x=0,y=1,hue="Cluster",data=df_agg_all_data ,palette =sns.color_palette("hls",2),ax=ax1)
ax1.set_title("Agglomerative model for All Data visualize in 2 dim (PCA) ")
plt.sca(ax2)
sns.scatterplot(x=0,y=1,hue="Class",data=df_agg_all_data,palette =sns.color_palette("hls",2),ax=ax2)
ax2.set_title("ALL Data after PCA dim reduce  ")



pd.crosstab(df_agg_all_data["Cluster"],df_agg_all_data['Class'])



print("score AGG model run to all data")
print_clustering_score(y_all_data,agg_cluster)
#print_silouhette_score(X_all_data,agg_cluster) 



#pd_data_pca.head()


# feature distribution in each cluster (after PCA)


print("Au revoir :)")

