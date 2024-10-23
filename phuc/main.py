
from collections import Counter
from joblib import Parallel, delayed
from matplotlib import cm
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
from similarity_graph import similarity_graph, polar_distance, LimitedSortedArray
import time

#Hàm chuyển Partition thành Clusters
def turn_to_clusters(partition: list):
    # Khởi tạo cluster là một từ điển rỗng
    clusters = {}

    # Cặp partition của Louvain bao gồm sessionID và số cụm tương ứng
    # Nếu số cụm chưa có trong thì tạo cho nó một mảng rỗng để thêm lần lượt các sesionID vào 
    for session_id, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(session_id)

    # Lọc ra những cụm chỉ chứa một phần tử duy nhất
    filtered_clusters = {cluster_id: members for cluster_id, members in clusters.items() if len(members) > 1}
    return filtered_clusters


# Hàm thực thi thuật toán Louvain
# Trả về partition, chỉ số modularity và chỉ số sihouette
def louvain_algorithm(graph: nx.Graph):
    # Phân cụm và có được partitions cũng như modularity
    partitions = community_louvain.best_partition(graph, resolution=1)
    modularity = community_louvain.modularity(partitions, graph)

    # Đối với sihouette, nếu chỉ có một cụm duy nhất thì không tính được Sihouette
    unique_clusters = set(partitions.values())
    if len(unique_clusters) > 1:
        silhouette = silhouette_score(nx.to_numpy_array(graph), list(partitions.values()), metric='euclidean')
    else:
        silhouette = None 
    return partitions, modularity, silhouette

# Hàm thực thi việc phân cụm cho đồ thị
def cluster(graph_data: similarity_graph, length: int):
    print("Start clustering...")

    # Khai báo những chỉ số cần thiết cho việc phân cụm trong đồ thị
    # Những giá trị partitions, modularity và sihouette tốt nhất mà việc phân cụm có thể có
    best_partitions = None
    best_modularity = -100
    best_sihouette = -100

    # Cụm tốt nhất, được tạo từ best_partitions
    best_clusters = None

    # Đồ thị lưu giá trị tốt nhất, dùng cho hiển thị đồ thị
    best_graph = None
    # Cụm được chia cho việc hiển thị đồ thị
    best_show_cluster = []


    # Cờ để dừng vòng lặp khi có giá trị tốt
    break_flag = False
    # Ngưỡng nhỏ nhất mà các cạnh cần phải lớn hơn để được thêm vào đồ thị
    threshold = 0
    # Biến đếm số lần lặp và số lần lặp tối đa
    cluster_loop_count = 0
    max_cluser_loop_count = 5

    # Giá trị k trong thuật toán KNN Graph (được lấy tổng số đỉnh chia cho 2)
    k = length // 2

    # Hàm lặp khi chưa có cờ và số lần lặp chưa đạt tối đa
    while cluster_loop_count <= max_cluser_loop_count and not break_flag:
        
        # Tăng biến đếm và thông báo cho người dùng
        cluster_loop_count += 1
        print(f"Cluster loop count: {cluster_loop_count}, k = {k} ")
        
        # Tạo đồ thị KNN Graph từ dữ liệu Similarity Graph (get knn graph)
        # Sau đó tiến hành bỏ những phần tử không liên thông (remove singleton)
        graph = graph_data.remove_singleton(graph_data.get_knn_graph(threshold, k))
        
        # Phân cụm đồ thị và lấy ra những giá trị tốt nhất
        partitions, modularity, silhouette = louvain_algorithm(graph)

        # In ra hai chỉ số tương ứng của lần lặp
        print(modularity)
        print(silhouette)

        # Nếu chỉ số modularity đạt mức chấp nhận được, lưu lại các giá trị
        if modularity > best_modularity:
            best_modularity = modularity
            best_sihouette = silhouette
            best_partitions = partitions 
            best_graph = graph


        # Nếu chỉ số modularity đạt mức tốt => Ngưng vòng lặp
        if modularity >= 0.3:
            break_flag = True
        else:
            # Ngược lại, giảm giá trị k để tăng modularity
            k //= 2
        
    # Chuyển partition thành cluster và cluster_show
    best_clusters = turn_to_clusters(best_partitions)
    for idx, cluster in best_clusters.items():
        best_show_cluster.append(cluster)

    print("Modularity: ", best_modularity)
    print("Silhouette: ", best_sihouette)
    
    return best_partitions, best_modularity, best_sihouette, best_clusters, best_show_cluster, best_graph
    


# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(1539).copy()

#
default_attributes = ['page 2 (clothing model)', 'colour', 'location', 'page 1 (main category)', 'page']


start_time = time.time()

n_grams = 3
main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)
for idx, attribute in enumerate(default_attributes):
    temp = data.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) >= n_grams)]
    main_grouped[attribute] = filtered_temp[attribute].apply(lambda x: list(ngrams(x, n_grams)))
    main_grouped['session ID'] = filtered_temp['session ID']


number_of_sessionID = main_grouped.shape[0]
number_of_train = int(number_of_sessionID * 0.9)
number_of_test = number_of_sessionID - number_of_train

train_grouped = main_grouped.head(n = number_of_train)
test_grouped = main_grouped.tail(n = number_of_test)

graph_data = similarity_graph(train_grouped, n_grams)
graph_data.write_adjacency_matrix()

graph_data.create_sample_graph(data)

best_attributes = ['colour', 'location', 'page 1 (main category)', 'page']

best_partitions, best_modularity, best_sihouette, best_clusters, best_show_cluster, best_graph = cluster(graph_data, number_of_train)

similarity_graph.show_graph(best_graph, best_show_cluster, showLegend = True, showEdges = False, showWeights = False)



def find_cluster(sessionID1, best_partitions, test_grouped, train_grouped, check_attributes, threshold = 0):
    list_neighbours = LimitedSortedArray(5)
    
    for sessionID2, clusterID in best_partitions.items():
        distance = 0
        for attribute in check_attributes:
            ngrams1 = test_grouped[test_grouped['session ID'] == sessionID1][attribute].values[0]
            ngrams2 = train_grouped[train_grouped['session ID'] == sessionID2][attribute].values[0]
            distance += 0.5 - polar_distance(ngrams1, ngrams2)
        distance /= float(len(check_attributes))
        if distance > threshold:
            list_neighbours.add([distance, clusterID])
    
    most_common_clusterID = Counter([neighbour[1] for neighbour in list_neighbours.data]).most_common(1)[0][0]
    return most_common_clusterID


clusters = best_clusters

def test_model(sessionID, best_attributes = best_attributes, best_partitions = best_partitions, data = data, test_grouped = test_grouped, train_grouped = train_grouped, clusters = clusters):
    check_attributes = best_attributes.copy()
    clusterID = find_cluster(sessionID, best_partitions, test_grouped, train_grouped, check_attributes)
    
    prediction_product = []
    product_list = list(data[data['session ID'] == sessionID]['page 2 (clothing model)'])

    one_third = len(product_list) // 3
    given_product_list =  product_list[:one_third]
    true_product_list = product_list[one_third:]

    for given_product in given_product_list:
        for node in clusters[clusterID]:
            product_id_list = train_grouped[train_grouped['session ID'] == node]['page 2 (clothing model)'].values[0]
            for product_id in product_id_list:
                if product_id[0] == given_product:
                    prediction_product.extend(product_id[1:])
                if len(prediction_product) >= len(true_product_list)*2:
                    break
            if len(prediction_product) >= len(true_product_list)*2:
                break
        if len(prediction_product) >= len(true_product_list)*2:
                break
        

    
    matching_elements = set(prediction_product) & set(true_product_list)
    matching_count = len(matching_elements)

    if len(prediction_product) != 0:
        precision = matching_count / len(prediction_product)
    else:
        precision = 0
    
    if matching_count != 0:
        recall = matching_count / len(true_product_list)
    else:
        recall = 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1_score

print("Testing model....")
results = Parallel(n_jobs=8, prefer="threads")(
    delayed(test_model)(sessionID) for sessionID in list(test_grouped['session ID'])
)
total_precision = 0
total_recall = 0
total_f1_score = 0
total_count = 0

for precision, recall, f1_score in results:
    total_precision += precision
    total_recall += recall
    total_f1_score += f1_score
    total_count += 1

print("Total precision: ", total_precision / total_count)
print("Total recall: ", total_recall / total_count)
print("Total F1 score: ", total_f1_score / total_count)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

    



