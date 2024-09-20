
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

# Hàm tạo đồ thị từ dữ liệu
def create_graph(table_data, distance_function, threshold, attributes):
    G = nx.Graph()


    for sessionID in table_data['session ID']:
        G.add_node(sessionID)

    for idx1, row1 in table_data.iterrows():
        session_id1 = row1['session ID']

        for idx2, row2 in table_data.iterrows():
            session_id2 = row2['session ID']
            if (session_id2 <= session_id1):
                continue
            sum = 0
            for attribute in attributes:
                clothing_models1 = tuple(row1[attribute])
                clothing_models2 = tuple(row2[attribute])
                sum += distance_function(clothing_models1, clothing_models2, 2)
            average_distance = sum / len(attributes)
            if average_distance < threshold:
                G.add_edge(session_id1, session_id2, weight=average_distance)                
                    
        
    return G

# Hàm gom cụm bằng thuật toán Louvain
def lovain_algorithm(G):
    # Sử dụng thuật toán Louvain để phân cụm
    partition = community_louvain.best_partition(G)

    clusters = {}

    for session_id, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(session_id)

    return clusters

# Hàm vẽ đồ thị
def show_graph(clusters, G, id, showLegend = True, showEdges = True):
    cmap = matplotlib.colormaps.get_cmap('tab20')

    color_map = {}
    color_index = 0

    # Tạo danh sách chú thích (legend)
    legend_items = []

    for cluster_id, nodes in clusters.items():
        if len(nodes) == 1:
            # Cụm chỉ có 1 phần tử -> gán màu xanh dương
            for node in nodes:
                color_map[node] = 'white'
        else:
            # Cụm có nhiều phần tử -> gán màu từ bảng màu
            current_color = cmap(color_index)
            for node in nodes:
                color_map[node] = current_color
            color_index += 1
            legend_items.append(mpatches.Patch(color=current_color, label=f"Cluster {cluster_id}"))  # Chú thích cho cụm có nhiều phần tử

    # Lấy danh sách màu cho các node trong đồ thị
    node_colors = [color_map[node] for node in G.nodes()]

    # Vẽ đồ thị
    pos = nx.spring_layout(G)  # Layout cho đồ thị
    
    if showEdges:
        nx.draw(G, pos, node_color=node_colors, node_size=50)
    else:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    # Thêm chú thích
    plt.legend(handles=legend_items, loc="upper right", title="Cluster")
    plt.title(f'Graph {id}')
    plt.show()

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

# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 5828)

# Nhóm dữ liệu theo session ID, nó trả về DataFrame bao gồm nhiều cột
# Từ DataFrame ta lấy cột clothing model, biến nó thành list bằng apply()
# Từ List, ta biến nó trở thành DataFrame
# grouped_clothingmodel = data.groupby('session ID')['page 2 (clothing model)'].apply(list).reset_index()

# Lọc dữ liệu, chọn những chuỗi có độ dài tối thiểu là 2
# filtered_grouped = grouped_clothingmodel[grouped_clothingmodel['page 2 (clothing model)'].apply(lambda x: len(x) > 2)]

# Danh sách thuộc
attributes = ['page 2 (clothing model)', 'colour', 'location', 'model photography', 'price 2', 'page']

columns_name = ['session ID'] + attributes 
grouped = pd.DataFrame(columns = columns_name)
for idx, attribute in enumerate(attributes):
    temp = data.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) > 2)]
    grouped[attribute] = filtered_temp[attribute]
    grouped['session ID'] = filtered_temp['session ID']


# Tạo đồ thị từ dữ liệu
G = create_graph(grouped, polar_distance, 0.5, attributes)

# # Tìm tất cả các thành phần liên thông
connected_components = list(nx.connected_components(G))

# # Lọc ra những bộ phận liên thông có nhiều hơn 2 đỉnh
filtered_components = [component for component in connected_components if len(component) > 2]

# # Tạo các đồ thị con từ mỗi thành phần liên thông
sub_graphs = [G.subgraph(component).copy() for component in filtered_components]


# # Đưa các đồ thị liên thông vào, tiến hành gom cụm cho từng đồ thị
for i, sub_graph in enumerate(sub_graphs):
    clusters = lovain_algorithm(sub_graph)
    show_graph(clusters, sub_graph, f"Big Graph {i}", showEdges=False)
    # for id, cluster in clusters.items():
    #     if len(cluster) > 1:
    #         sub_graph1 = sub_graph.subgraph(cluster).copy()
    #         clusters1 = lovain_algorithm(sub_graph1)
    #         show_graph(clusters1, sub_graph1, f"Graph {id}", showEdges=False)

