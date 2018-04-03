# Example code for generating a clustered heatmap. 

import pandas
import numpy
import seaborn
import re
import scipy
import matplotlib

data = pandas.read_csv('Gene_40.csv',index_col='Unnamed: 0')

#Cleaning up columnames for aesthetic reasons
#Also important to get the color bars working
m = list(data)
celltypes = [re.sub(r'\.\d+','',i)for i in m]
newcols = dict(zip(m,celltypes))
data.rename(columns=newcols,inplace = True)
#code to produce a color bar identifying each cluster. If you run this code
#a labeled white bar will appear. I am still not sure why the colors won't show
#would be nice to have.
# m2 = list(data)
# m2 = pandas.DataFrame({'cell_type':m2})
# lut = dict(zip(m2.cell_type.unique(),"rbgcmky"))
# col_colors = m2.cell_type.map(lut)

#Additional options for the cluster.
#figsize=(30,10)
#,col_colors = col_colors
#Clusters and draws the heatmap
x = seaborn.clustermap(data,method = 'average',metric='euclidean',row_cluster=True,col_cluster=True)

#saves image to file
x.savefig("clustered heatmap.png",dpi=600)
matplotlib.pyplot.close()
#xticklables=t

#Dendrogram
#using pearson's correlation
correlations = data.corr()
correlations_array = numpy.asarray(data.corr())
row_linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.pdist(correlations_array), method='average')
col_linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.pdist(correlations_array.T), method='average')
scipy.cluster.hierarchy.dendrogram(col_linkage,labels=list(data))
matplotlib.pyplot.close()

#using euclidean distance
col_link_euc = scipy.cluster.hierarchy.linkage(data.transpose(),method="average",metric="euclidean")
fig_size = matplotlib.pyplot.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 10
matplotlib.pyplot.rcParams["figure.figsize"] = fig_size

#matplotlib.pyplot.figsize((50,15))
den_euc = scipy.cluster.hierarchy.dendrogram(col_link_euc,labels=list(data))


#mathplotlib.pyplot.savefig("Euclidean dendrogram.png")
matplotlib.pyplot.gcf()
matplotlib.pyplot.savefig("Euclidean dendrogram.png",dpi=600)

matplotlib.pyplot.close()
