# import leidenalg
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from community import community_louvain
# from nltk import ngrams
# import numpy as np
# import igraph as ig

# # Hàm để tính khoảng cách polar giữa hai chuỗi hành động
# def polar_distance(list1, list2, n):
#     # Tìm danh sách n-grams từ hai chuỗi ban đầu
#     ngrams_list1 = list(ngrams(list1, n))
#     ngrams_list2 = list(ngrams(list2, n))

#     # Tìm phần hợp giữa hai danh sách
#     union_ngrams = ngrams_list1 + ngrams_list2

#     # Tạo hai danh sách đại diện cho hai dãy tần suất chuẩn hóa tương ứng hai danh sách n-grams
#     # Ta xem mỗi danh sách giống như một vectơ
#     frequency_vector1 = []
#     frequency_vector2 = []

#     # Tìm tần suất chuẩn hóa của các phần tử trong phần hợp so hai danh sách n-grams
#     for ngram in union_ngrams:
#         # Đếm số lần xuất hiện của từng phần tử trong phần hợp
#         # Chia số lần xuất hiện cho số lượng phần tử của mỗi danh sách
#         frequency_vector1.append(ngrams_list1.count(ngram)/ len(ngrams_list1))
#         frequency_vector2.append(ngrams_list2.count(ngram) / len(ngrams_list2))

#     #  distance = 1/pi x arccos(tích vô hướng hai vector / (độ dài vectơ 1 * độ dài vectơ 2))

#     # Tính tích vô hướng hai vectơ

#     dot_product = sum(x * y for x, y in zip(frequency_vector1, frequency_vector2))

#     # Tính độ dài của vectơ 1
#     magnitude_f1 = np.sqrt(sum(x ** 2 for x in frequency_vector1))

#     # Tính độ dài của vectơ 2
#     magnitude_f2 = np.sqrt(sum(x ** 2 for x in frequency_vector2))

#     # Tìm độ tương đồng cosin
#     cosine_similarity = dot_product / (magnitude_f1 * magnitude_f2)

#     # Giới hạn giá trị cosin [-1, 1]
#     cosine_similarity = max(min(cosine_similarity, 1), -1)

#     # Tìm khoảng cách polar
#     polar_distance_result = (1 / np.pi) * np.arccos(cosine_similarity)
#     return polar_distance_result

# # Đọc dữ liệu từ CSV
# df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# # Lấy ra n dòng đầu tiên từ dữ liệu cho trước
# data = df.head(n = 5000)


# # Nhóm dữ liệu theo session ID, nó trả về DataFrame bao gồm nhiều cột
# # Từ DataFrame ta lấy cột clothing model, biến nó thành list bằng apply()
# # Từ List, ta biến nó trở thành DataFrame
# grouped = data.groupby('session ID')['page 2 (clothing model)'].apply(list).reset_index()

# # Lọc dữ liệu, chọn những chuỗi có độ dài tối thiểu là 2
# filtered_grouped = grouped[grouped['page 2 (clothing model)'].apply(lambda x: len(x) > 2)]

# # Tạo đồ thị từ dữ liệu
# G = nx.Graph()
# a = grouped['page 2 (clothing model)']

# for idx, row in filtered_grouped.iterrows():
#     session_id = row['session ID']
#     clothing_models = tuple(row['page 2 (clothing model)'])
#     G.add_node(session_id, value = tuple(clothing_models))

# for idx1, row1 in filtered_grouped.iterrows():
#     session_id1 = row1['session ID']
#     clothing_models1 = tuple(row1['page 2 (clothing model)'])
#     for idx2, row2 in filtered_grouped.iterrows():
#         session_id2 = row2['session ID']
#         clothing_models2 = tuple(row2['page 2 (clothing model)'])
#         if session_id2 > session_id1:
#             distance = polar_distance(clothing_models1, clothing_models2, 2)
#             if distance < 0.37:
#                 G.add_edge(session_id1, session_id2, weight = distance)

# for node in G.nodes:
#     G.nodes[node]["name"] = node
# # Chuyển đổi NetworkX Graph thành igraph Graph
# G_igraph = ig.Graph.from_networkx(G)

# # for u, v, weight in G.edges(data='weight'):
# #     print(f"Cạnh giữa {u} và {v} có trọng số: {weight}")
# # Sử dụng thuật toán Leiden để phân cụm
# # partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
# partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition,weights="weight")

# # Hiển thị các cụm
# for community_id, community in enumerate(partition):
#     original_names = [G_igraph.vs[node]["name"] for node in community]
#     print(f"Cluster {community_id}: {sorted(original_names)}")

# # Tạo dictionary để gán màu cho các node
# color_map = {}

# # available_colors = ['black', 'green', 'yellow', 'orange', 'purple', 'pink']
# available_colors = [
#     'black', 'green', 'yellow', 'orange', 'purple', 'pink', 'red', 
#     'brown', 'cyan', 'magenta', 'lime', 'olive', 'navy', 'teal', 'coral', 
#     'turquoise', 'salmon', 'beige', 'gold', 'indigo', 'lavender', 'violet', 
#     'khaki', 'plum', 'orchid', 'silver', 'tan', 'wheat', 'tomato', 'slateblue', 
#     'lightgreen', 'darkred',  'darkgreen', 'darkorange', 
#     'lightcoral',  'hotpink', 'chartreuse', 'seagreen'
# ]
# color_index = 0

# # Duyệt qua từng cụm
# for community_id, community in enumerate(partition):
#     if len(community) == 1:
#         # Nếu cụm chỉ có một phần tử, gán màu xanh dương
#         for node in community:
#             original_node = G_igraph.vs[node]["name"]  # Lấy lại tên đỉnh ban đầu
#             color_map[original_node] = 'blue'
#     else:
#         # Nếu cụm có nhiều phần tử, gán màu mới từ available_colors
#         for node in community:
#             original_node = G_igraph.vs[node]["name"]  # Lấy lại tên đỉnh ban đầu
#             color_map[original_node] = available_colors[color_index % len(available_colors)]
#         color_index += 1

# # for community_id, community in enumerate(partition):
# #     color_map[community[0]] = "blue"

# # print(color_map)
# # # Vẽ đồ thị với màu sắc cho các node
# # # node_colors = [color_map[node] for node in G.nodes()]
# node_colors = [color_map.get(node, 'red') for node in G.nodes]

# # node_colors = [color_map[node] for node in G.nodes()]

# # pos = nx.spring_layout(G)
# pos = nx.spring_layout(G)
#  # Thay vì spring_layout
# # labels = {node: f'{node}' for node in G.nodes}
# labels = {node: str(node) for node in G.nodes()}
# nx.draw_networkx_nodes(G, pos, label=labels,node_color=node_colors, node_size=30)


# plt.show()






##############################################
from collections import Counter
import leidenalg
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from nltk import ngrams
import numpy as np
import igraph as ig


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

# Đọc dữ liệu từ CSV
df = pd.read_csv('mapping_data.csv')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 3533)


# Nhóm dữ liệu theo session ID, nó trả về DataFrame bao gồm nhiều cột
# Từ DataFrame ta lấy cột clothing model, biến nó thành list bằng apply()
# Từ List, ta biến nó trở thành DataFrame
grouped = data.groupby('session ID')['page 2 (clothing model)'].apply(list).reset_index()

# Lọc dữ liệu, chọn những chuỗi có độ dài tối thiểu là 2
filtered_grouped = grouped[grouped['page 2 (clothing model)'].apply(lambda x: len(x) > 2)]

# Tạo đồ thị từ dữ liệu
G = nx.Graph()
a = grouped['page 2 (clothing model)']

for idx, row in filtered_grouped.iterrows():
    session_id = row['session ID']
    clothing_models = tuple(row['page 2 (clothing model)'])
    G.add_node(session_id, value = tuple(clothing_models))

for idx1, row1 in filtered_grouped.iterrows():
    session_id1 = row1['session ID']
    clothing_models1 = tuple(row1['page 2 (clothing model)'])
    for idx2, row2 in filtered_grouped.iterrows():
        session_id2 = row2['session ID']
        clothing_models2 = tuple(row2['page 2 (clothing model)'])
        if session_id2 > session_id1:
            distance = polar_distance(clothing_models1, clothing_models2, 2)
            if distance < 0.37:
                G.add_edge(session_id1, session_id2, weight = distance)

for node in G.nodes:
    G.nodes[node]["name"] = node
# Chuyển đổi NetworkX Graph thành igraph Graph
G_igraph = ig.Graph.from_networkx(G)

# for u, v, weight in G.edges(data='weight'):
#     print(f"Cạnh giữa {u} và {v} có trọng số: {weight}")
# Sử dụng thuật toán Leiden để phân cụm
# partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition,weights="weight")

partition = [community for community in old if len(community) > 1]

# for community in old:
#     if len(community) == 1:
#         # Xóa nút ra khỏi đồ thị
#         node_to_remove = G_igraph.vs[community[0]]["name"]
#         print(node_to_remove)
#         G.remove_node(node_to_remove)

for community_id, community in enumerate(partition):
    original_names = [G_igraph.vs[node]["name"] for node in community]
    print(f"Cluster {community_id}: {sorted(original_names)}")

# Tạo dictionary để gán màu cho các node
color_map = {}

# available_colors = ['black', 'green', 'yellow', 'orange', 'purple', 'pink']
available_colors = [
    'black', 'green', 'yellow', 'orange', 'purple', 'pink', 
    'brown', 'cyan', 'magenta', 'lime', 'olive', 'navy', 'teal', 'coral', 
    'turquoise', 'salmon', 'beige', 'gold', 'indigo', 'lavender', 'violet', 
    'khaki', 'plum', 'orchid', 'silver', 'tan', 'wheat', 'tomato', 'slateblue', 
    'lightgreen',   'darkgreen', 'darkorange', 
    'lightcoral',  'hotpink', 'chartreuse', 'seagreen'
]
color_index = 0

# Duyệt qua từng cụm
for community_id, community in enumerate(partition):
    if len(community) == 1:
        # Nếu cụm chỉ có một phần tử, gán màu xanh dương
        for node in community:
            original_node = G_igraph.vs[node]["name"]  # Lấy lại tên đỉnh ban đầu
            color_map[original_node] = 'blue'
    else:
        # Nếu cụm có nhiều phần tử, gán màu mới từ available_colors
        for node in community:
            original_node = G_igraph.vs[node]["name"]  # Lấy lại tên đỉnh ban đầu
            color_map[original_node] = available_colors[color_index % len(available_colors)]
        color_index += 1

# for community_id, community in enumerate(partition):
#     color_map[community[0]] = "blue"

# print(color_map)
# # Vẽ đồ thị với màu sắc cho các node
# # node_colors = [color_map[node] for node in G.nodes()]

def get_most_common_attribute(community, attribute, df):
    # Lấy ra các session ID trong cụm
    session_ids = [G_igraph.vs[node]["name"] for node in community]

    # Lọc dữ liệu theo các session ID thuộc cụm hiện tại
    filtered_data = df[df['session ID'].isin(session_ids)]

    # Đếm tần suất xuất hiện của từng giá trị trong thuộc tính
    counter = Counter(filtered_data[attribute])
    # Trả về giá trị phổ biến nhất
    most_common = counter.most_common(1)[0]
    result = f"Giá trị: {most_common[0]}, Tan suat: {most_common[1]}"
    return result
cluster_labels = {}

for community_id, community in enumerate(partition):
    # Tìm thuộc tính phổ biến nhất trong cột 'page 1 (main category)' và 'colour'
    most_common_color = get_most_common_attribute(community, 'colour', data)
    most_common_category = get_most_common_attribute(community, 'page 1 (main category)', data)
    most_common_price = get_most_common_attribute(community, 'price', data)
    most_common_product = get_most_common_attribute(community, 'page 2 (clothing model)', data)

    # Gán nhãn cho cụm với thuộc tính phổ biến nhất
    cluster_labels[community_id] = f"Color: {most_common_color}, Category: {most_common_category},Price : {most_common_price},Product: {most_common_product},"
    


for community_id, label in cluster_labels.items():
    print(f"Cluster {community_id}: {label}")

node_colors = [color_map.get(node, 'red') for node in G.nodes]
pos = nx.spring_layout(G)
 # Thay vì spring_layout
# labels = {node: f'{node}' for node in G.nodes}
labels = {node: str(node) for node in G.nodes()}
nx.draw_networkx_nodes(G, pos, label=labels,node_color=node_colors, node_size=30)


plt.show()

