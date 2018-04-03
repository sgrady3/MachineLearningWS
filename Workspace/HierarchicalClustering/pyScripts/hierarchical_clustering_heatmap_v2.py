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

data = pandas.read_csv('/home/stephen/code/MachineLearningWS/Data/GeneExpression/Gene_40.csv',index_col='Unnamed: 0')

#Cleaning up columnames for aesthetic reasons
m = list(data)
celltypes = [re.sub(r'\.\d+','',i)for i in m]
newcols = dict(zip(m,celltypes))
data.rename(columns=newcols,inplace = True)

# m2 = list(data)
# m2 = pandas.DataFrame({'cell_type':m2})
# m2 = m2.pop('cell_type')
# lut = dict(zip(m2.unique(),"rbgcmky"))
# col_colors = m2.map(lut)
# 
# 
# lut = dict(zip(data.columns.unique(),"rbgcmky"))
# col_colors = m2.cell_type.map(lut)
# 
# 

# col_colors = pandas.DataFrame(m2.cell_type.map(lut))
# col_colors = col_colors.transpose()
# newcols = dict(zip(list(col_colors),celltypes))
# col_colors.rename(columns=newcols,inplace = True)
# data2 = data
# 
# x = seaborn.clustermap(data.reset_index(drop=True),method = 'average',robust=True,metric='euclidean',z_score=0,row_cluster=True,col_cluster=True,col_colors=col_colors)
# 
# x.savefig("clustered heatmap.png",dpi=600)
# 
# x = seaborn.clustermap(data,col_colors=col_colors,figsize=(2,2))
# x.savefig("clustered heatmap.png",dpi=600)


# #######################
# # 1. default settings #
# #######################
# 
# # defaults: 
# # method = average, metric = euclidean (most commonly used), row_cluster=True, col_cluster=True
# 
# hm_default = seaborn.clustermap(data)
# 
# #saves image to file
# hm_default.savefig("output/1_genes_40_clustered_default.png",dpi=300)
# matplotlib.pyplot.close()
# 
# ####################
# # 2. rotate labels #
# ####################
# 
# hm_labels = seaborn.clustermap(data)
# 
# plt.setp(hm_labels.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_labels.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# #saves image to file
# hm_labels.savefig("output/2_genes_40_clustered_labels.png",dpi=300)
# matplotlib.pyplot.close()
# 
# ####################
# # 3. change colors #
# ####################
# 
# hm_colors = seaborn.clustermap(data, cmap="Oranges")
# 
# plt.setp(hm_colors.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_colors.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# #saves image to file
# hm_colors.savefig("output/3_genes_40_clustered_colors.png",dpi=300)
# matplotlib.pyplot.close()
# 
# #######################
# # 4. method = average #
# #######################
# 
# hm_average = seaborn.clustermap(data,method = 'average',metric='euclidean', cmap="Oranges")
# 
# plt.setp(hm_average.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_average.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# hm_average.savefig("output/4_genes_40_clustered_average.png",dpi=300)
# matplotlib.pyplot.close()
# 
# 
# ########################
# # 5. method = complete #
# ########################
# 
# hm_complete = seaborn.clustermap(data,method = 'complete',metric='euclidean', cmap="Blues")
# 
# plt.setp(hm_complete.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_complete.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# hm_complete.savefig("output/5_genes_40_clustered_complete.png",dpi=300)
# matplotlib.pyplot.close()
# 
# ######################
# # 6. method = single #
# ######################
# 
# hm_single = seaborn.clustermap(data,method = 'single',metric='euclidean', cmap="Greens")
# 
# plt.setp(hm_single.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_single.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# hm_single.savefig("output/6_genes_40_clustered_single.png",dpi=300)
# matplotlib.pyplot.close()
# 
# ########################
# # 7. method = centroid #
# ########################
# 
# hm_centroid = seaborn.clustermap(data,method = 'centroid',metric='euclidean', cmap="Purples")
# 
# plt.setp(hm_centroid.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_centroid.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# hm_centroid.savefig("output/7_genes_40_clustered_centroid.png",dpi=300)
# matplotlib.pyplot.close()
# 
# #####################
# # 8. larger dataset #
# #####################
# 
# data = pandas.read_csv('input/Gene_2000.csv',index_col='Unnamed: 0')
# 
# m = list(data)
# celltypes = [re.sub(r'\.\d+','',i)for i in m]
# newcols = dict(zip(m,celltypes))
# data.rename(columns=newcols,inplace = True)
# 
# hm_genes_2000_complete = seaborn.clustermap(data,method = 'complete',metric='euclidean', cmap="Blues")
# 
# plt.setp(hm_genes_2000_complete.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
# plt.setp(hm_genes_2000_complete.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# 
# hm_genes_2000_complete.savefig("output/8_genes_2000_clustered_complete.png",dpi=300)
# matplotlib.pyplot.close()

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



# #using euclidean distance, centroid linkage doesn't work yet, # of cluster not correct
# col_link_euc_cent = link(data.transpose(),method="centroid",metric="euclidean")
# fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
# fig_size[0] = 20
# fig_size[1] = 10
# matplotlib.pyplot.rcParams["figure.figsize"] = fig_size
# 
# den_euc = dend(col_link_euc_cent,labels=list(data))
# 
# plt.gcf()
# plt.savefig("output/Euclidean dendrogram_cent.png",dpi=600)
# plt.close('all')
# 
# #Cut the dendrogram
# three_clusters = cut(col_link_euc_cent,n_clusters=3)
# 
# #Calculate Silhouette metric
# #Overall score for all the data
# s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# #Scores for each data point
# s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# 
# #Plot Silhouette
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_cent")
# 
# #compare different # of clusters
# four_clusters = cut(col_link_euc_cent,n_clusters=4)
# five_clusters = cut(col_link_euc_cent,n_clusters=5)
# six_clusters = cut(col_link_euc_cent,n_clusters=6)
# seven_clusters = cut(col_link_euc_cent,n_clusters=7)
# ten_clusters = cut(col_link_euc_cent,n_clusters=10)
# 
# s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
# s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
# s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
# s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
# s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))
# 
# 
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouette_cent")
# Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_cent")
# Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_cent")
# Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_cent")
# Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_cent")
# 
# #using euclidean distance, weighted linkage
# col_link_euc_weighted = link(data.transpose(),method="weighted",metric="euclidean")
# fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
# fig_size[0] = 20
# fig_size[1] = 10
# matplotlib.pyplot.rcParams["figure.figsize"] = fig_size
# 
# den_euc = dend(col_link_euc_weighted,labels=list(data))
# 
# plt.gcf()
# plt.savefig("output/Euclidean dendrogram_weighted.png",dpi=600)
# plt.close('all')
# 
# #Cut the dendrogram
# three_clusters = cut(col_link_euc_weighted,n_clusters=3)
# 
# #Calculate Silhouette metric
# #Overall score for all the data
# s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# #Scores for each data point
# s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# 
# #Plot Silhouette
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_weighted")
# 
# #compare different # of clusters
# four_clusters = cut(col_link_euc_weighted,n_clusters=4)
# five_clusters = cut(col_link_euc_weighted,n_clusters=5)
# six_clusters = cut(col_link_euc_weighted,n_clusters=6)
# seven_clusters = cut(col_link_euc_weighted,n_clusters=7)
# ten_clusters = cut(col_link_euc_weighted,n_clusters=10)
# 
# s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
# s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
# s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
# s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
# s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))
# 
# 
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouette_weighted")
# Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_weighted")
# Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_weighted")
# Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_weighted")
# Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_weighted")
# 
# #using euclidean distance, median
# col_link_euc_median = link(data.transpose(),method="median",metric="euclidean")
# fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
# fig_size[0] = 20
# fig_size[1] = 10
# matplotlib.pyplot.rcParams["figure.figsize"] = fig_size
# 
# den_euc = dend(col_link_euc_median,labels=list(data))
# 
# plt.gcf()
# plt.savefig("output/Euclidean dendrogram_median.png",dpi=600)
# plt.close('all')
# 
# #Cut the dendrogram
# three_clusters = cut(col_link_euc_median,n_clusters=3)
# 
# #Calculate Silhouette metric
# #Overall score for all the data
# s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# #Scores for each data point
# s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
# 
# #Plot Silhouette
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_median")
# 
# #compare different # of clusters
# four_clusters = cut(col_link_euc_median,n_clusters=4)
# five_clusters = cut(col_link_euc_median,n_clusters=5)
# six_clusters = cut(col_link_euc_median,n_clusters=6)
# seven_clusters = cut(col_link_euc_median,n_clusters=7)
# ten_clusters = cut(col_link_euc_median,n_clusters=10)
# 
# s_sample_four = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(four_clusters))
# s_sample_five = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(five_clusters))
# s_sample_six = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(six_clusters))
# s_sample_seven = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(seven_clusters))
# s_sample_ten = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(ten_clusters))
# 
# 
# 
# Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouette_median")
# Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_median")
# Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_median")
# Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_median")
# Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_median")





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

print(4)


#using euclidean distance, ward linkage looks best, best # of clusters is 4
d=data.transpose()
col_link_euc_ward = link(preprocessing.scale(d),method="ward",metric="euclidean")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

den_euc = dend(col_link_euc_ward,labels=list(data))

plt.gcf()
plt.savefig("output/Euclidean dendrogram_ward_scaled.png",dpi=600)
plt.close('all')

#Cut the dendrogram
three_clusters = cut(col_link_euc_ward,n_clusters=3)

#Calculate Silhouette metric
#Overall score for all the data
s_ave_three = sk_s_score(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))
#Scores for each data point
s_sample_three = sk_sample(data.transpose(),metric="euclidean",labels=np.ravel(three_clusters))

#Plot Silhouette

Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_ward_scaled")
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



Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouette_ward_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_ward_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_ward_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_ward_scaled")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_ward_scaled")
plt.close('all')

#create gene lists from clusters
c0 = [index for index, item in enumerate(three_clusters) if item==0]
c1 = [index for index, item in enumerate(three_clusters) if item==1]
c2 = [index for index, item in enumerate(three_clusters) if item==2]

cluster0 = data.iloc[:,c0]
cluster1 = data.iloc[:,c1]
cluster2 = data.iloc[:,c2]

print(5)

#using correlation distance, 
col_link_euc = link(data.transpose(),method="average",metric="correlation")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

den_euc = dend(col_link_euc,labels=list(data))

plt.gcf()
plt.savefig("output/correlation dendrogram_avg.png",dpi=600)
plt.close('all')

#Cut the dendrogram
three_clusters = cut(col_link_euc,n_clusters=3)

#Calculate Silhouette metric
#Overall score for all the data
s_ave_three = sk_s_score(data.transpose(),metric="correlation",labels=np.ravel(three_clusters))
#Scores for each data point
s_sample_three = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(three_clusters))

#Plot Silhouette

Silhouette_plot.s_plot(data.transpose(),s_sample_three,three_clusters,show_plot=False,save_dir="output/Three_cluster_Silhouette_avg_cor")
plt.close('all')
#compare different # of clusters
four_clusters = cut(col_link_euc,n_clusters=4)
five_clusters = cut(col_link_euc,n_clusters=5)
six_clusters = cut(col_link_euc,n_clusters=6)
seven_clusters = cut(col_link_euc,n_clusters=7)
ten_clusters = cut(col_link_euc,n_clusters=10)

s_sample_four = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(four_clusters))
s_sample_five = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(five_clusters))
s_sample_six = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(six_clusters))
s_sample_seven = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(seven_clusters))
s_sample_ten = sk_sample(data.transpose(),metric="correlation",labels=np.ravel(ten_clusters))



Silhouette_plot.s_plot(data.transpose(),s_sample_four,four_clusters,show_plot=False,save_dir="output/Four_cluster_Silhouett_avg_cor")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_five,five_clusters,show_plot=False,save_dir="output/Five_cluster_Silhouette_avg_cor")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_six,six_clusters,show_plot=False,save_dir="output/Six_cluster_Silhouette_avg_cor")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_seven,seven_clusters,show_plot=False,save_dir="output/Seven_cluster_Silhouette_avg_cor")
plt.close('all')
Silhouette_plot.s_plot(data.transpose(),s_sample_ten,ten_clusters,show_plot=False,save_dir="output/Ten_cluster_Silhouette_avg_cor")
plt.close('all')