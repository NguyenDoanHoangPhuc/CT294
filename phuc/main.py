
import pprint
from matplotlib import cm
import matplotlib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from networkx.algorithms.community import girvan_newman
from nltk import ngrams
import numpy as np
import matplotlib.patches as mpatches
from pyvis.network import Network
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score

from similarity_graph import similarity_graph

#Hàm chuyển Partition thành Clusters
def turn_to_clusters(partition):
    clusters = {}

    for session_id, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(session_id)

    filtered_clusters = {cluster_id: members for cluster_id, members in clusters.items() if len(members) > 1}
    return filtered_clusters

def analyze_cluster(cluster, dataset):
    # Lọc các dòng trong dataset có sessionID thuộc cụm
    filtered_data = dataset[dataset['session ID'].isin(cluster)]

    # Các thuộc tính cần phân tích
    attributes = ['page 1 (main category)', 'colour', 
                  'location', 'model photography', 'price 2', 'page']
    
    # Tạo danh sách để lưu kết quả
    analysis_results = []
    labels = []

    # Duyệt qua từng thuộc tính để tính giá trị chiếm nhiều nhất và phần trăm
    for attribute in attributes:
        value_counts = filtered_data[attribute].value_counts()
        
        if not value_counts.empty:
            # Tìm giá trị chiếm nhiều nhất và phần trăm của nó
            most_frequent_value = value_counts.idxmax()
            percentage = (value_counts.max() / len(filtered_data)) * 100
            
            if percentage >= 50:
                labels.append(label(attribute, most_frequent_value))
            # Lưu kết quả
            analysis_results.append({
                'Attribute': attribute,
                'Most Frequent Value': most_frequent_value,
                'Percentage': f"{percentage:.2f}%"
            })
    
    # Chuyển kết quả thành DataFrame để hiển thị dưới dạng bảng
    results_df = pd.DataFrame(analysis_results)
    
    # In bảng kết quả
    print("Labels of the cluster: \n", labels)
    print(results_df)
    return labels

def label(attribute, value):
    if attribute == 'page 1 (main category)':
        name = ['trousers', 'skirts', 'blouses', 'sale']
        return 'Category ' + name[value-1]
    elif attribute == 'colour':
        name = ['beige', 'black', 'blue', 'brown', 'burgundy', 'gray', 'green', 'navy blue', 'of many colors', 'olive', 'pink', 'red', 'violet', 'white']
        return 'Color ' + name[value-1]
    elif attribute == 'location':
        name = ['top left', 'top in the middle', 'top right', 'bottom left', 'bottom in the middle', 'bottom right']
        return 'Location ' + name[value-1]
    elif attribute == 'model photography':
        name = ['en-face', 'profile']
        return 'Model Photography ' + name[value-1]
    elif attribute == 'price 2':
        if value == 1:
            return 'Higher price product'
        else:
            return 'No higher price product'
    elif attribute == 'page':
        return 'Page ' + str(value)
    else:
        return 'Unknown attribute'

def clustering(graph):
    partitions = community_louvain.best_partition(graph, randomize=True, resolution=1)
    modularity = community_louvain.modularity(partitions, graph)

    return partitions, modularity

def remove_attribute(attributes, graph, data):
    
    if len(attributes) == 1:
        return False
    
    sessionID_list = list(graph.nodes())
    filtered_df = data[data['session ID'].isin(sessionID_list)]
    filtered_columns_df = filtered_df.loc[:, 'page 1 (main category)': 'page']
    filtered_columns_df['cluster'] = data['session ID'].map(partitions)
                
    flag = False
    for attribute in attributes:
        contingency = pd.crosstab(filtered_columns_df['cluster'], filtered_columns_df[attribute])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        if p_value > 0.05:
             flag = True
             attributes.remove(attribute)

    if len(attributes) == 0:
        return False
    return flag              

# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 30620)

default_attributes = ['page 1 (main category)','page 2 (clothing model)', 'colour', 'location', 'model photography', 'price 2', 'page']

result = []

n_grams = 5
main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)
for idx, attribute in enumerate(default_attributes):
    temp = data.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) >= n_grams)]
    main_grouped[attribute] = filtered_temp[attribute]
    main_grouped['session ID'] = filtered_temp['session ID']

graph_data = similarity_graph(main_grouped, n_grams)

graph_data.create_sample_graph(data)
clusters_result = []


# Tao nhung thuoc tinh can thiet cho qua trinh phan cum
threshold = 0
attributes = default_attributes.copy()
# Bien kiem soat xem vong lap phan cum chay bao nhieu lan
cluster_loop_count = 0
# Co bien kiem soat de thoat vong lap
break_flag = False

best_partitions = None
best_modularity = -100
best_graph = None

minimum_divistion = 0.01
max_cluser_loop_count = 0.5 / minimum_divistion 

while cluster_loop_count < max_cluser_loop_count and not break_flag:
    cluster_loop_count += 1
    
    # Tu bang grouped, ta tao duoc do thi tuong dong tu lop SimilarityGraph
    graph = graph_data.remove_singleton(graph_data.get_knn_graph(attributes, threshold, 10))
    
    partitions, modularity = clustering(graph)

    if modularity > best_modularity:
        best_modularity = modularity
        best_partitions = partitions
        best_graph = graph

    # Neu chi so modularity phu hop thi them vao nodes_list2, ket thuc vong lap
    if modularity >= 0.5:
        best_graph = graph
        clusters = turn_to_clusters(partitions)
        print("Modularity: ", modularity)
        for cluster_id, cluster in clusters.items():
            clusters_result.append(cluster)
        break_flag = True
    else:
        removed = remove_attribute(attributes, graph, data)
        if removed == False:
            threshold += minimum_divistion
            attributes = default_attributes.copy()     

if not break_flag:
    print("Modularity: ", best_modularity)
    clusters = turn_to_clusters(best_partitions)
    for cluster_id, cluster in clusters.items():
        clusters_result.append(cluster)

 
similarity_graph.show_graph(best_graph, clusters_result, showLegend = True, showEdges = False, showWeights = False)
