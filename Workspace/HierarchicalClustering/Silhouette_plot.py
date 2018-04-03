import numpy as np
from sklearn.metrics import silhouette_score as sk_s_score
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def s_plot(Data,silhouette_scores,clusters,show_plot=True,save_dir=False,dpi=150):
#plots silhouettes calculated using scikit learns sample silhouette metric
#Code taken from sci kit learn 
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# and  adapted into a function
    plt.ioff()
    clusters = np.ravel(clusters)
    n_clusters = max(clusters)+1
    y_lower=10
    plt.close('all')
    plt.ioff()
    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(20,15)
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            silhouette_scores[clusters == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    silhouette_avg = sk_s_score(Data,metric="euclidean",labels=np.ravel(clusters))
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ticks = [-1,-0.8,-0.6,-0.4,-0.2,0, 0.2, 0.4, 0.6, 0.8, 1]
    if min(silhouette_scores)<0:
        for i in range(1,5):
            if min(silhouette_scores)<ticks[i]:
                ax1.set_xticks(ticks[i-1:11])
                break
    else:
        ax1.set_xticks([-0.1,0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.rcParams.update({'font.size': 18})
    if show_plot==True:
        plt.ion()
        plt.show()
    if save_dir!=False:
        fig.savefig(save_dir+".png",dpi=dpi)
    plt.ion()
    return;
