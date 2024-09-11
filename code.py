import leidenalg
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from networkx.algorithms.community import girvan_newman
from nltk import ngrams
import numpy as np

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
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 3533)

# Nhóm dữ liệu theo session ID, nó trả về DataFrame bao gồm nhiều cột
# Từ DataFrame ta lấy cột clothing model, biến nó thành list bằng apply()
# Từ List, ta biến nó trở thành DataFrame
grouped = data.groupby('session ID')['page 2 (clothing model)'].apply(list).reset_index()

# Lọc dữ liệu, chọn những chuỗi có độ dài tối thiểu là 2
filtered_grouped = grouped[grouped['page 2 (clothing model)'].apply(lambda x: len(x) > 2)]

G = nx.Graph()

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
            if distance < 0.5:
                G.add_edge(session_id1, session_id2, weight = distance)

# Sử dụng thuật toán Girvan-Newman để phân cụm
comp = girvan_newman(G)
clusters = next(comp)
# Hiển thị các cụm
for communities in next(comp):
    print(f"Cluster: {sorted(communities)}")

# # Sử dụng thuật toán Louvain để phân cụm
# partition = community_louvain.best_partition(G, weight='weight', resolution=0.8)
#
# # Hiển thị các cụm
# for session_id, cluster_id in partition.items():
#     print(f"Session ID {session_id} thuộc về cụm {cluster_id}")

# Tạo dictionary để gán màu cho các node
color_map = {}

# Khởi tạo danh sách các màu (bỏ qua xanh dương vì nó dành cho các cụm có một phần tử)
available_colors = ['red', 'green', 'yellow', 'orange', 'purple', 'pink']
color_index = 0

# Duyệt qua từng cụm
for cluster in clusters:
    if len(cluster) == 1:
        # Nếu cụm chỉ có một phần tử, gán màu xanh dương
        for node in cluster:
            color_map[node] = 'blue'
    else:
        # Nếu cụm có nhiều phần tử, gán màu khác (theo thứ tự từ danh sách màu)
        current_color = available_colors[color_index % len(available_colors)]
        for node in cluster:
            color_map[node] = current_color
        color_index += 1

# Lấy danh sách màu cho các node để vẽ đồ thị
node_colors = [color_map[node] for node in G.nodes()]

# Vẽ đồ thị với màu sắc đã gán
pos = nx.spring_layout(G)  # Layout cho đồ thị
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)

# Hiển thị đồ thị
plt.show()



