
import math
import pickle
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# 读取存储的邻接矩阵
def load_graphs_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        adjacency_matrices = pickle.load(f)
    return adjacency_matrices


# 将邻接矩阵转换为NetworkX图对象
def convert_to_graphs(adjacency_matrices):
    graphs = [nx.from_numpy_array(matrix) for matrix in adjacency_matrices]
    return graphs


# 计算图特征
def compute_graph_features(graph):
    degrees = [d for n, d in graph.degree()]#度
    clustering_coeff = list(nx.clustering(graph).values())
    # eigenvector_centrality = list(nx.eigenvector_centrality_numpy(graph).values())#特征向量中心性
    pagerank = list(nx.pagerank(graph).values())
    betweenness_centrality = list(nx.betweenness_centrality(graph).values())

    # 计算特征的平均值
    features = [
        np.mean(degrees),
        np.mean(clustering_coeff),
        # np.mean(eigenvector_centrality),
        np.mean(pagerank),
        np.mean(betweenness_centrality)
    ]
    return features


# 使用肘部法估计最优簇数量
def estimate_optimal_clusters_elbow(features, max_clusters=8):
    sse = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()

    # 找到“肘部”点
    elbow_point = np.diff(sse, 2).argmin() + 2  # 二阶差分的最小值位置
    return elbow_point


# 使用轮廓系数估计最优簇数量
def estimate_optimal_clusters_silhouette(features, max_clusters=8):
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal Clusters')
    plt.show()

    # 找到轮廓系数最高的点
    best_n_clusters = np.argmax(silhouette_scores) + 2
    return best_n_clusters


# 主函数
def main1(file):
    # file='result/256'
    file_path = f'{file}/graphs.pkl'  # 修改为你的pkl文件路径
    f = open(file_path, 'rb')
    data = pickle.load(f)
    adjacency_matrices = load_graphs_from_pkl(file_path)
    n = int(math.sqrt(len(adjacency_matrices)))
    # adjacency_matrices=[value for value in adjacency_matrices.values()]

    graphs = convert_to_graphs(adjacency_matrices)
    features = np.array([compute_graph_features(g) for g in graphs])

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 使用肘部法估计最优簇数量
    optimal_clusters_elbow = estimate_optimal_clusters_elbow(scaled_features)
    print(f"Optimal number of clusters (Elbow Method): {optimal_clusters_elbow}")

    # 使用轮廓系数估计最优簇数量
    optimal_clusters_silhouette = estimate_optimal_clusters_silhouette(scaled_features)
    print(f"Optimal number of clusters (Silhouette Method): {optimal_clusters_silhouette}")

    # 选择其中一个最优簇数量进行聚类
    # n_clusters = optimal_clusters_silhouette  # 这里选择轮廓系数方法的结果
    # n_clusters = 4 # 这里选择轮廓系数方法的结果
    n_clusters = optimal_clusters_elbow  # 这里选择肘部法的结果
    labels, kmeans = cluster_graphs(scaled_features, n_clusters)
    lable=labels.reshape(n,n)
    # 输出聚类结果
    for i, label in enumerate(labels):
        print(f"Graph {i} is in cluster {label}")

    # 可选：将结果保存到文件中
    result_df = pd.DataFrame({'Graph': list(range(len(labels))), 'Cluster': labels})
    with open(f'{file}/clustering_results.pkl', 'wb') as f:
        pickle.dump(lable, f)
    # result_df.to_csv(f'{file}/clustering_results1.csv', index=False)
#

def cluster_graphs(features, n_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    return labels, kmeans



# main1('result/bert/wh/128')
# main1('result/wh/128/pos-neg')