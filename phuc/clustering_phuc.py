
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

from SimilarityGraph import SimilarityGraph

# Hàm để tính khoảng cách polar giữa hai chuỗi hành động
def polar_distance(list1, list2, n):
    # Tìm danh sách n-grams từ hai chuỗi ban đầu
    ngrams_list1 = list(ngrams(list1, n))
    ngrams_list2 = list(ngrams(list2, n))

    # Tìm phần hợp giữa hai danh sách
    union_ngrams = ngrams_list1 + ngrams_list2

    # Tạo hai danh sách đại diện cho hai dãy tần suất chuẩn hóa tương ứng hai danh sách n-grams
    # Ta xem mỗi danh sách giống như một vectơ
    frequency_vector1 = []
    frequency_vector2 = []

    # Tìm tần suất chuẩn hóa của các phần tử trong phần hợp so hai danh sách n-grams
    for ngram in union_ngrams:
        # Đếm số lần xuất hiện của từng phần tử trong phần hợp
        # Chia số lần xuất hiện cho số lượng phần tử của mỗi danh sách
        frequency_vector1.append(ngrams_list1.count(ngram)/ len(ngrams_list1))
        frequency_vector2.append(ngrams_list2.count(ngram) / len(ngrams_list2))

    #  distance = 1/pi x arccos(tích vô hướng hai vector / (độ dài vectơ 1 * độ dài vectơ 2))

    # Tính tích vô hướng hai vectơ

    dot_product = sum(x * y for x, y in zip(frequency_vector1, frequency_vector2))

    # Tính độ dài của vectơ 1
    magnitude_f1 = np.sqrt(sum(x ** 2 for x in frequency_vector1))

    # Tính độ dài của vectơ 2
    magnitude_f2 = np.sqrt(sum(x ** 2 for x in frequency_vector2))

    # Tìm độ tương đồng cosin
    cosine_similarity = dot_product / (magnitude_f1 * magnitude_f2)

    # Giới hạn giá trị cosin [-1, 1]
    cosine_similarity = max(min(cosine_similarity, 1), -1)

    # Tìm khoảng cách polar
    polar_distance_result = (1 / np.pi) * np.arccos(cosine_similarity)
    return polar_distance_result

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
    partitions = community_louvain.best_partition(graph)
    labels = [partitions[node] for node in graph.nodes()]
    
    if len(set(labels)) >= 2:  # Make sure there are at least 2 clusters
        silhouette = silhouette_score(nx.to_numpy_array(graph), labels)
    else:
        silhouette = -1  # Invalid silhouette score if only 1 cluster exists

    return partitions, silhouette

def remove_attribute(attributes, graph, data):
    sessionID_list = list(graph.nodes())
    filtered_df = data[data['session ID'].isin(sessionID_list)]
    filtered_columns_df = filtered_df.loc[:, 'page 1 (main category)': 'page']
    filtered_columns_df['cluster'] = data['session ID'].map(partitions)
                
    for attribute in attributes:
        contingency = pd.crosstab(filtered_columns_df['cluster'], filtered_columns_df[attribute])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        if p_value > 0.05 and len(attributes) > 1:
             attributes.remove(attribute)
    return attributes                 

# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 2038)

default_attributes = ['page 2 (clothing model)', 'colour', 'location', 'model photography', 'price 2', 'page']

main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)
for idx, attribute in enumerate(default_attributes):
    temp = data.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) > 2)]
    main_grouped[attribute] = filtered_temp[attribute]
    main_grouped['session ID'] = filtered_temp['session ID']

sGraph = SimilarityGraph(main_grouped, 0, default_attributes, polar_distance)


nodes_list = []
nodes_list.append(data['session ID'].tolist())
time_count = 0

while (time_count <= 0):
    nodes_list2 = []
    print("Time count: ", time_count)

    for nodes in nodes_list:
        # Tao nhung thuoc tinh can thiet cho qua trinh phan cum
        threshold = 0
        attributes = default_attributes.copy()
        # Bien kiem soat xem vong lap phan cum chay bao nhieu lan
        cluster_loop_count = 0
        # Co bien kiem soat de thoat vong lap
        break_flag = False
        
        best_silhouette = -100
        best_partitions = None

        while cluster_loop_count < 80 and not break_flag:
            cluster_loop_count += 1
            # Tao bang du lieu tu dataset da duoc loc
            columns_name = ['session ID'] + attributes
            grouped = pd.DataFrame(columns = columns_name)

            # Them du lieu vao bang grouped
            for idx, attribute in enumerate(attributes):
                temp = data.groupby('session ID')[attribute].apply(list).reset_index()
                temp = temp[temp['session ID'].isin(nodes)]
                filtered_temp = temp[temp[attribute].apply(lambda x: len(x) > 2)]
                grouped[attribute] = filtered_temp[attribute]
                grouped['session ID'] = filtered_temp['session ID']

            # Tu bang grouped, ta tao duoc do thi tuong dong tu lop SimilarityGraph
            similarityGraph = SimilarityGraph(grouped, threshold, attributes, polar_distance)

            result = clustering(similarityGraph.graph)
            partitions = result[0]
            silhouette_louvain = result[1]

            if silhouette_louvain > best_silhouette:
                best_silhouette = silhouette_louvain
                best_partitions = partitions

            # Neu chi so silhouette phu hop thi them vao nodes_list2, ket thuc vong lap
            if silhouette_louvain >= 0.7:
               
                clusters = turn_to_clusters(partitions)
                for cluster_id, cluster in clusters.items():
                    nodes_list2.append(cluster)
                break_flag = True
            # Neu chi so silhouette khong phu hop, ma khong co thuoc tinh nao de loai thi ta giam nguong threshold
            elif len(attributes) <= 1:
                threshold += 0.05
                attributes = default_attributes.copy()
            # Neu chi so silhouette khong phu hop, ta loai bo thuoc tinh khong phu hop
            else:
                removed = remove_attribute(attributes, similarityGraph.graph, data)
                if len(removed) == 0:
                    threshold += 0.05
                    attributes = default_attributes.copy()     
        if not break_flag:
            clusters = turn_to_clusters(best_partitions)
            for cluster_id, cluster in clusters.items():
                nodes_list2.append(cluster)
    
    nodes_list = nodes_list2.copy()
    time_count += 1



sGraph.show_graph(nodes_list, showLegend = True, showEdges = False, showWeights=False)
