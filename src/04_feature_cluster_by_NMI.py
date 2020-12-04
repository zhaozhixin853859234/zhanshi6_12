import pandas as pd
import numpy as np
import json
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import spectral_clustering
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
np.set_printoptions(suppress=True, threshold=np.nan)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# 用不同聚类模型对特征进行聚类
def try_different_cluster_model(model,data,num_cluster):
    model.fit(data)
    labels = model.predict(data)
    print(namestr(model, globals())[0], "模型聚类标签：\n",labels)
    # print(labels)
    # print(sum(labels))

    # 两个聚类评价指标
    silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(data, labels)
    # adjusted_rand_score = metrics.adjusted_rand_score(data, labels)
    print(namestr(model, globals())[0], "模型的轮廓系数为：",silhouette_score)
    print(namestr(model, globals())[0], "模型的Calinski-Harabasz 值为：", calinski_harabaz_score)
    # print(namestr(model, globals())[0], "模型的调整兰德系数为：", adjusted_rand_score)

    # 画图展示，二维图中展示前两个特征,labels表示聚类后类别
    for i in range(1, len(labels)):
        if labels[i] == 0:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='red')
        elif labels[i] == 1:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='blue')
        elif labels[i] == 2:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='green')
        elif labels[i] == 3:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='cyan')
        elif labels[i] == 4:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='magenta')
        elif labels[i] == 5:
            plt.scatter(df_array[i, 0], df_array[i, 1], s=15, c='black')
    title_name = str(namestr(model, globals())[0])+str("---")+str(num_cluster)+str("---")+str(" result")
    plt.title(title_name)
    plt.xlabel('x')
    plt.ylabel('y')
    fig_savePath = str("../data/featureClusterResult/")+str(namestr(model, globals())[0])+str("--")+str(num_cluster)+str("-cluster")+str(".png")
    plt.savefig(fig_savePath)
    plt.close()

    # 把特征所在的分类存储
    cluster_result = dict(zip(list(range(len(labels))), list(labels)))

    cluster_result_groupby = dict(zip(list(range(num_cluster)), [[] for i in range(num_cluster)]))
    for key, value in cluster_result.items():
        cluster_result_groupby[value].append(key)
    # 转换字典值
    cluster_result_groupby1 = {}
    for key in cluster_result_groupby.keys():
        try:
            #print("key:",key)
            a = key+1
            #print("a:",a)
            key_reset = str("V")+str(a)
            cluster_result_groupby[key_reset] = cluster_result_groupby.pop(key)
            #print(cluster_result_groupby)
        except:
            break
    # print(cluster_result)
    # print(cluster_result_groupby)
    return cluster_result_groupby


if __name__ == '__main__':
    df = pd.read_csv("../data/experiment_data/MI_info_withLabel.csv", header=None)

    # df = df.drop(df.shape[0]-1, axis=1)
    print(df.shape)
    df_array = df.values
    num_cluster = 3

    # K-均值聚类
    kmeans_model = KMeans(n_clusters=num_cluster)
    viewConstruction_res = try_different_cluster_model(kmeans_model, df_array, num_cluster)
    # 将视图存为json文件
    jsObj = json.dumps(viewConstruction_res)
    view_save_path = str("../data/featureClusterResult/K-means_Cluster/")+str(namestr(kmeans_model, globals())[0])+str("--")+str(num_cluster)+\
                     str("-cluster_group_discretion.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

    # 高斯混合聚类
    gmmModel = GaussianMixture(n_components=num_cluster, covariance_type='diag', random_state=0)
    viewConstruction_res = try_different_cluster_model(gmmModel, df_array, num_cluster)
    # 将视图存为json文件
    jsObj = json.dumps(viewConstruction_res)
    view_save_path = str("../data/featureClusterResult/GMM_Cluster/")+str(namestr(gmmModel, globals())[0])+str("--")+str(num_cluster)+\
                     str("-cluster_group_discretion.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

    # # 均值漂移聚类
    # bandwidth = estimate_bandwidth(df_array, quantile=0.2, n_samples=300)
    # meanShiftModel = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # try_different_cluster_model(meanShiftModel, df_array, num_cluster)

    # 谱聚类
    # spectral_clusteringModel = spectral_clustering(n_clusters=num_cluster, eigen_solver='arpack',affinity=df_array)
    # print(spectral_clusteringModel)

    # 层次聚类





