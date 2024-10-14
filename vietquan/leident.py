
from collections import Counter
import leidenalg
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from nltk import ngrams
import numpy as np
import igraph as ig
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
import community
from joblib import Parallel, delayed

import time
start_time = time.time()

# n_grams = 5 # threshold = 0.09 - 30000
n_grams = 4 # threshold = 0 - 30000
max_length = 10
# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
dataFrame = df.head(n = 40000)
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
    # print(polar_distance_result)
    return polar_distance_result


def create_matrix(table_data, attributes):
    num_nodes = table_data['session ID'].max()
    
    # Tạo từ điển mặc định cho các thuộc tính
    def create_default_dict():
        return {attribute: 0 for attribute in attributes}

    # Tạo ma trận với các từ điển mặc định
    matrix = [[create_default_dict() for _ in range(num_nodes)] for _ in range(num_nodes)]

    def compute_edge(idx1, row1):
        session_id1 = row1['session ID']
        local_edges = []
        for idx2, row2 in table_data.iterrows():
            session_id2 = row2['session ID']
            if session_id2 > session_id1:
                edge_dict = {}
                for attribute in attributes:
                    clothing_models1 = tuple(row1[attribute])
                    clothing_models2 = tuple(row2[attribute])
                    edge_dict[attribute] = 0.5 - polar_distance(clothing_models1, clothing_models2, n_grams)  # 2 là 2 grams
                local_edges.append((session_id1, session_id2, edge_dict))
        return local_edges

    # Gọi Parallel để chạy song song
    results = Parallel(n_jobs=-1)(
        delayed(compute_edge)(idx1, row1) for idx1, row1 in table_data.iterrows()
    )

    # Cập nhật ma trận từ kết quả
    for local_edges in results:
        for session_id1, session_id2, edge_dict in local_edges:
            matrix[session_id1 - 1][session_id2 - 1] = edge_dict
            matrix[session_id2 - 1][session_id1 - 1] = edge_dict

    return matrix






default_attributes = ['page 2 (clothing model)', 'colour', 'location', 'model photography', 'price 2', 'page']
main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)
for idx, attribute in enumerate(default_attributes):
    temp = dataFrame.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) >= n_grams)]
    main_grouped[attribute] = filtered_temp[attribute]
    main_grouped['session ID'] = filtered_temp['session ID']

matrix = create_matrix(main_grouped,default_attributes )
print("create matrix done.")
main_graph = nx.Graph()
for sessionID in dataFrame['session ID']:
   main_graph.add_node(sessionID)
   main_graph.nodes[sessionID]['name'] = sessionID


def remove_attribute(attributes, graph):
    sessionID_list = list(graph.nodes())
    filtered_df = dataFrame[dataFrame['session ID'].isin(sessionID_list)]
    filtered_columns_df = filtered_df.loc[:, 'page 1 (main category)': 'page']
                

    G_igraph = ig.Graph.from_networkx(graph)
    session_to_cluster = {}
    for cluster_id, cluster in enumerate(partitions):
        for session_id in cluster:
            # Get the original session ID
            original_session_id = G_igraph.vs[session_id]["name"]
            session_to_cluster[original_session_id] = cluster_id
    filtered_columns_df['cluster'] = df['session ID'].map(session_to_cluster)

    hasRemove = 0
    for attribute in attributes:
        contingency = pd.crosstab(filtered_columns_df['cluster'], filtered_columns_df[attribute])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        if p_value > 0.05 and len(attributes) > 1:
             attributes.remove(attribute)
             hasRemove = 1
    return attributes ,hasRemove               



def add( data,value):
        if len(data) >= max_length:
            if (value[0] > data[0][0]):
                data.pop(0)
            else:
                return
        # Add the new value to the array
        data.append(value)
        # Sort the array after each insertion
        data.sort()

# Hàm tạo đồ thị từ dữ liệu
# def create_matrix(table_data,attributes ):
#         num_nodes =  table_data['session ID'].max()
#         default_dict = {attribute: 0 for attribute in attributes}
#         matrix = [[default_dict.copy() for _ in range(num_nodes)] for _ in range(num_nodes)]

#         for idx1, row1 in table_data.iterrows():
#             session_id1 = row1['session ID']
#             for idx2, row2 in table_data.iterrows():
#                 session_id2 = row2['session ID']
#                 if (session_id2 > session_id1):
#                     edge_dict = {}
#                     for attribute in attributes:
#                         clothing_models1 = tuple(row1[attribute])
#                         clothing_models2 = tuple(row2[attribute])
#                         edge_dict[attribute] = 0.5 - polar_distance(clothing_models1, clothing_models2, 2) #2 la 2 grams
#                     matrix[session_id1-1][session_id2-1] = edge_dict
#                     matrix[session_id2-1][session_id1-1] = edge_dict
    
#         return matrix


import time
def create_graph(main_graph ,matrix, attributes, threshold):
        # chưa có đồ thị
        # limited_arr = []
        # start_time = time.time()
        # for i in range(100000):
        #     add(limited_arr,[1, 2])

        # end_time = time.time()
        # print(f"Thời gian thực thi: {end_time -start_time :.6f} giây")
        G = main_graph.copy()
    
        for idx1, row1 in main_grouped.iterrows():
            session_id1 = row1['session ID']
            limited_arr = []
            for idx2, row2 in main_grouped.iterrows():
                session_id2 = row2['session ID']
                if (session_id2 > session_id1):
                    sum = 0
                    for attribute in attributes:
                        sum += matrix[session_id1-1][session_id2-1][attribute]

                    average_distance = sum / len(attributes)
                    if average_distance > threshold:
                        add(limited_arr,[average_distance, session_id2])
            for item in limited_arr:
                G.add_edge(session_id1, item[1], weight = item[0])   
        print(G)
        return G



# def create_graph(main_graph, data, matrix, attributes, threshold):
#     # Sao chép đồ thị chính
#     G = main_graph.copy()

#     def compute_edges_for_session(index1, row1):
#         session_id1 = row1['session ID']
#         limited_arr = []
#         print(session_id1)
#         for idx2, row2 in data.iterrows():
#             session_id2 = row2['session ID']
#             if session_id2 > session_id1:
#                 sum_distances = sum(matrix[session_id1 - 1][session_id2 - 1][attribute] for attribute in attributes)
#                 average_distance = sum_distances / len(attributes)
#                 if average_distance > threshold:
#                     add(limited_arr, [average_distance, session_id2,session_id1])
#         return limited_arr

#     # Gọi Parallel để chạy song song
#     results = Parallel(n_jobs=-1)(
#         delayed(compute_edges_for_session)(idx1, row1)
#         for idx1, row1 in data.iterrows()
#     )

#     # Cập nhật đồ thị sau khi có tất cả các kết quả
#     for session_edges in results:
#         for average_distance, session_id2,session_id1 in session_edges:
#             G.add_edge(session_id1, session_id2, weight=average_distance)

#     return G


# Hàm vẽ đồ thị
def show_graph(clusters, G,id, showLegend = True, showEdges = True):
    cmap = matplotlib.colormaps.get_cmap('tab20')
    color_map = {}
    color_index = 0

    # Tạo danh sách chú thích (legend)
    legend_items = []
    G_igraph = ig.Graph.from_networkx(G)

    
    for cluster_id, nodes in enumerate(clusters):
        if len(nodes) == 1:
            # Cụm chỉ có 1 phần tử -> gán màu xanh dương
            for node in nodes:
                original_node = G_igraph.vs[node]["name"]
                color_map[original_node] = 'white'
        else:
            # Cụm có nhiều phần tử -> gán màu từ bảng màu
            current_color = cmap(color_index)
            for node in nodes:
                original_node = G_igraph.vs[node]["name"]
                color_map[original_node] = current_color
            color_index += 1
            legend_items.append(mpatches.Patch(color=current_color, label=f"Cluster {cluster_id}"))  # Chú thích cho cụm có nhiều phần tử

    # Lấy danh sách màu cho các node trong đồ thị
    node_colors = [color_map[node] for node in G.nodes()]

    # Vẽ đồ thị
    pos = nx.spring_layout(G)  # Layout cho đồ thị
      # Gán nhãn cho các node là tên của node
    # labels = {node: node for node in G.nodes()}  # Tạo từ điển nhãn, node chính là tên của node
    if showEdges:
        nx.draw(G, pos, node_color=node_colors, node_size=10)
    else:
        # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=10)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=10)
    # Thêm chú thích
    # nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black') 
    plt.legend(handles=legend_items, loc="upper right", title="Cluster")
    plt.title(f'Graph {id}')
    plt.show()

def analyze_cluster(cluster, dataset):
    # Lọc các dòng trong dataset có sessionID thuộc cụm
    filtered_data = dataset[dataset['session ID'].isin(cluster)]

    # Các thuộc tính cần phân tích
    attributes = ['page 1 (main category)', 'colour', 
                  'location', 'model photography', 'price 2', 'page']
    # attributes = ['page 2 (clothing model)', 'page']
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
    G_igraph = ig.Graph.from_networkx(graph)
    partitions = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition,weights="weight")
    labels = partitions.membership
    modularity_score = partitions.modularity
    # if len(set(labels)) >= 2:  # Make sure there are at least 2 clusters
    #     silhouette = silhouette_score(nx.to_numpy_array(graph), labels)
    # else:
    #     silhouette = -1  # Invalid silhouette score if only 1 cluster exists

    # print("Giá trị modularity:", modularity_score)
    return partitions, modularity_score

def remove_singleton( graph):
        connected_components = list(nx.connected_components(graph))
    
        # Iterate over each connected component
        for component in connected_components:
            # If the component has only one node, remove it from the graph
            if len(component) == 1:
                graph.remove_node(next(iter(component)))  # Get the single node and remove it
        
        return graph


            # Them du lieu vao bang grouped




# nodes_list = []
# nodes_list.append(data['session ID'].tolist())
# time_count = 0
# partitions = 0


# while (time_count <= 0):
#     nodes_list2 = []
#     # print("Time count: ", time_count)

#     for nodes in nodes_list:
#         # Tao nhung thuoc tinh can thiet cho qua trinh phan cum
#         threshold = 0
#         attributes = default_attributes.copy()
#         # Bien kiem soat xem vong lap phan cum chay bao nhieu lan
#         cluster_loop_count = 0
#         # Co bien kiem soat de thoat vong lap
#         break_flag = False
        
#         best_silhouette = -100
#         best_partitions = None

#         while cluster_loop_count < 80 and not break_flag:
#             cluster_loop_count += 1
#             # Tao bang du lieu tu dataset da duoc loc
#             columns_name = ['session ID'] + attributes
#             grouped = pd.DataFrame(columns = columns_name)

#             # Them du lieu vao bang grouped
#             for idx, attribute in enumerate(attributes):
#                 temp = data.groupby('session ID')[attribute].apply(list).reset_index()
#                 temp = temp[temp['session ID'].isin(nodes)]
#                 filtered_temp = temp[temp[attribute].apply(lambda x: len(x) > 2)]
#                 grouped[attribute] = filtered_temp[attribute]
#                 grouped['session ID'] = filtered_temp['session ID']

#             # Tu bang grouped, ta tao duoc do thi tuong dong tu lop SimilarityGraph
#             similarityGraph =  remove_singleton(create_graph(main_graph,grouped,matrix, attributes, threshold))
#             num_edges = similarityGraph.number_of_edges()
#             if num_edges == 0:
#                 break_flag = True
#                 break

#             # node_with_max_degree(similarityGraph)
#             for node in similarityGraph.nodes:
#                  similarityGraph.nodes[node]["name"] = node

#             result = clustering(similarityGraph)
#             partitions = result[0]
#             silhouette_leiden = result[1]

#             print(threshold, len(attributes),silhouette_leiden)
            
#             if silhouette_leiden > best_silhouette:
#                 best_silhouette = silhouette_leiden
#                 best_partitions = partitions
#                 best_graph = similarityGraph
#             # Neu chi so silhouette phu hop thi them vao nodes_list2, ket thuc vong lap
#             if silhouette_leiden >= 0.7:
#                 break_flag = True
#             # Neu chi so silhouette khong phu hop, ma khong co thuoc tinh nao de loai thi ta giam nguong threshold
#             elif len(attributes) <= 1:
#                 threshold += 0.01
#                 attributes = default_attributes.copy()
#             # Neu chi so silhouette khong phu hop, ta loai bo thuoc tinh khong phu hop
#             else:
#                 response = remove_attribute(attributes, similarityGraph, data)
#                 removed = response[0]
#                 hasRemove = response[1]
#                 # if(len(attribute) )
#                 if not hasRemove:
#                     threshold += 0.01
#                 if len(removed) == 0:
#                     threshold += 0.01
#                     attributes = default_attributes.copy() 

#         print(cluster_loop_count, best_silhouette)    
#     nodes_list = nodes_list2.copy()
#     time_count += 1



# Giả sử bạn đã định nghĩa time_count ở nơi khác trong mã
# nodes_list2 = []
# print("Time count: ", time_count)

# Giả sử nodes_list đã được định nghĩa ở nơi khác trong mã
# Thay vì vòng lặp for, bạn có thể xác định các node ở đây
# nodes_list = ...

threshold = 0
attributes = default_attributes.copy()
# Biến kiểm soát xem vòng lặp phân cụm chạy bao nhiêu lần
cluster_loop_count = 0
# Có biến kiểm soát để thoát vòng lặp
break_flag = False

best_silhouette = -100
best_partitions = None
best_graph = None
while cluster_loop_count < 80 and not break_flag:
    cluster_loop_count += 1
    threshold = round(threshold, 2)
    # Tu bang grouped, ta tao duoc do thi tuong dong tu lop SimilarityGraph
    # start_time = time.time()
    graph =  remove_singleton(create_graph(main_graph,matrix, attributes, threshold))
    # end_time = time.time()
    # print(graph)
    # print(f"Thời gian thực thi: {end_time - start_time :.6f} giây") 
    num_edges = graph.number_of_edges()
    if num_edges == 0:
        break_flag = True
        break
    partitions, modularity = clustering(graph)
    if modularity > best_silhouette:
       best_silhouette = modularity
       best_partitions = partitions
       best_graph = graph
    # Neu chi so modularity phu hop thi them vao nodes_list2, ket thuc vong lap
    if modularity >= 0.5:
        break_flag = True
    else:
        response = remove_attribute(attributes, graph)
        if response[1] == False:
            threshold += 0.01
            attributes = default_attributes.copy()     

    print("Lập:",cluster_loop_count, best_silhouette, threshold , " số thuộc tính ", len(attributes))    

print(threshold)
print(partitions)
end_time = time.time()
print(f"Thời gian thực thi: {end_time - start_time :.6f} giây") 
show_graph(partitions, best_graph, f"Big Graph ", showEdges=False)
