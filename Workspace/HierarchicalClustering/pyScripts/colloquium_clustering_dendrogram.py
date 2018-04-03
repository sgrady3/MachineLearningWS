# Example code for generating a clustered heatmap. 

import pandas
import numpy as np
import seaborn
import re
from scipy.cluster.hierarchy import linkage as link
from scipy.cluster.hierarchy import dendrogram as dend
from scipy.cluster.hierarchy import cut_tree as cut
from sklearn.metrics import silhouette_samples as sk_sample
from sklearn.metrics import silhouette_score as sk_s_score
from sklearn import preprocessing
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import Silhouette_plot

data = pandas.read_csv('input/Gene_40.csv',index_col='Unnamed: 0')

#Cleaning up columnames for aesthetic reasons
m = list(data)
celltypes = [re.sub(r'\.\d+','',i)for i in m]
newcols = dict(zip(m,celltypes))
data.rename(columns=newcols,inplace = True)


##################
# 9. Dendrogram #
##################

# defaults: 
# method = single, metric = euclidean (most commonly used)
plt.ioff()
print(1)
#using euclidean distance, average linkage works but has many "clusters" with single leaf
col_link_euc = link(data.transpose(),method="average",metric="euclidean")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

den_euc = dend(col_link_euc,labels=list(data))

plt.gcf()
plt.savefig("output/Euclidean dendrogram_avg.png",dpi=600)
plt.close('all')

#Cut the dendrogram
three_clusters = cut(col_link_euc,n_clusters=3)

#Calculate Silhouette metric
#Overall score for all the data
s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
#Scores for each data point
s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))

#Plot Silhouette

Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_avg")
plt.close('all')
#compare different # of clusters
four_clusters = cut(col_link_euc,n_clusters=4)
five_clusters = cut(col_link_euc,n_clusters=5)
six_clusters = cut(col_link_euc,n_clusters=6)
seven_clusters = cut(col_link_euc,n_clusters=7)
ten_clusters = cut(col_link_euc,n_clusters=10)

s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))



Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouett_avg")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_avg")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_avg")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_avg")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_avg")
plt.close('all')

print(2)


#using euclidean distance, ward linkage looks best, best # of clusters is 4
col_link_euc_ward = link(data.transpose(),method="ward",metric="euclidean")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

den_euc = dend(col_link_euc_ward,labels=list(data))

plt.gcf()
plt.savefig("output/Euclidean dendrogram_ward.png",dpi=600)
plt.close('all')

#Cut the dendrogram
three_clusters = cut(col_link_euc_ward,n_clusters=3)

#Calculate Silhouette metric
#Overall score for all the data
s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
#Scores for each data point
s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))

#Plot Silhouette

Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_ward")
plt.close('all')
#compare different # of clusters
four_clusters = cut(col_link_euc_ward,n_clusters=4)
five_clusters = cut(col_link_euc_ward,n_clusters=5)
six_clusters = cut(col_link_euc_ward,n_clusters=6)
seven_clusters = cut(col_link_euc_ward,n_clusters=7)
ten_clusters = cut(col_link_euc_ward,n_clusters=10)

s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))



Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouette_ward")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_ward")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_ward")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_ward")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_ward")
plt.close('all')

#create gene lists from clusters
c0 = [index for index, item in enumerate(three_clusters) if item==0]
c1 = [index for index, item in enumerate(three_clusters) if item==1]
c2 = [index for index, item in enumerate(three_clusters) if item==2]

cluster0 = data.iloc[:,c0]
cluster1 = data.iloc[:,c1]
cluster2 = data.iloc[:,c2]



## Scaling data
print(3)

#using euclidean distance, average linkage works but has many "clusters" with single leaf
d = data.transpose()
col_link_euc = link(preprocessing.scale(d),method="average",metric="euclidean")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

den_euc = dend(col_link_euc,labels=list(data))

plt.gcf()
plt.savefig("output/Euclidean dendrogram_avg_scaled.png",dpi=600)
plt.close('all')

#Cut the dendrogram
three_clusters = cut(col_link_euc,n_clusters=3)

#Calculate Silhouette metric
#Overall score for all the data
s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
#Scores for each data point
s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))

#Plot Silhouette

Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_avg_scaled")
plt.close('all')
#compare different # of clusters
four_clusters = cut(col_link_euc,n_clusters=4)
five_clusters = cut(col_link_euc,n_clusters=5)
six_clusters = cut(col_link_euc,n_clusters=6)
seven_clusters = cut(col_link_euc,n_clusters=7)
ten_clusters = cut(col_link_euc,n_clusters=10)

s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))



Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouett_avg_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_avg_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_avg_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_avg_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_avg_scaled")
plt.close('all')
