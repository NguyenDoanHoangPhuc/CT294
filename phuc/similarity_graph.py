import pprint
from joblib import Parallel, delayed
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import numpy as np
from nltk import ngrams

import threading
from multiprocessing import Process


# Hàm để tính khoảng cách polar giữa hai chuỗi hành động
def polar_distance(ngrams_list1, ngrams_list2):
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

    # distance = 1/pi x arccos(tích vô hướng hai vector / (độ dài vectơ 1 * độ dài vectơ 2))

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

class LimitedSortedArray:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []

    def add(self, value):
        if len(self.data) >= self.max_length:
            if (value[0] > self.data[0][0]):
                self.data.pop(0)
            else:
                return
        # Add the new value to the array
        self.data.append(value)
        # Sort the array after each insertion
        self.data.sort()

    def __repr__(self):
        return f"{self.data}"

class similarity_graph:
    def __init__(self, table_data, n_grams):
        self.n_grams = n_grams
        self.table_data = table_data
        self.num_nodes = self.get_num_nodes()
        self.graph = nx.Graph()
        self.adj_matrices = {}
        
    def get_num_nodes(self):
        return self.table_data['session ID'].max()

    def write_adjacency_matrix(self):
        n = self.num_nodes  
        table_data = self.table_data
        attributes = table_data.columns[2:]
        adj_matrix = np.memmap(f'adj_matrix.dat', dtype='float32', mode='w+', shape=(n, n))

        def compute_edge(idx1, row1):
            session_id1 = row1['session ID']
            local_edges = []
            for idx2, row2 in table_data.iterrows():
                session_id2 = row2['session ID']
                if session_id2 > session_id1:
                    avg = 0
                    for attribute in attributes:
                        clothing_models1 = tuple(row1[attribute])
                        clothing_models2 = tuple(row2[attribute])
                        avg += 0.5 - polar_distance(clothing_models1, clothing_models2)
                    avg /= len(attributes)
                    local_edges.append((session_id1, session_id2, avg))
            return local_edges

        print("Writing data...")
        results = Parallel(n_jobs=-1)(
            delayed(compute_edge)(idx1, row1) for idx1, row1 in table_data.iterrows()
        )

        print("Updating adjacency matrix...")
        # Cập nhật ma trận từ kết quả
        for local_edges in results:
            for session_id1, session_id2, value in local_edges:
                adj_matrix[session_id1 - 1, session_id2 - 1] = value
        
        print("Finished writing data!")
        
    def print_adjacency_matrix(self):
        n = self.num_nodes
        adj_matrix = np.memmap('adj_matrix.dat', dtype='float32', mode='r', shape=(n, n))
        
        for i in range(n):
            for j in range(n):
                print(f"{adj_matrix[i, j]:.2f}", end=" ")
            print()


    def create_sample_graph(self,data):
        for sessionID in data['session ID']:
            self.graph.add_node(sessionID)

    def get_knn_graph(self, threshold, k):

        data = self.table_data
        n = self.num_nodes
        G = self.graph.copy()

        adj_matrix = np.memmap('adj_matrix.dat', dtype='float32', mode='r', shape=(n, n))

        for idx1, row1 in data.iterrows():
            session_id1 = row1['session ID']
            limited_arr = LimitedSortedArray(k)
            for idx2, row2 in data.iterrows():
                session_id2 = row2['session ID']
                if (session_id2 > session_id1):
                    average_distance = adj_matrix[session_id1 - 1, session_id2 - 1]
                    if average_distance > threshold:
                        limited_arr.add([average_distance, session_id2])

            for item in limited_arr.data:
                G.add_edge(session_id1, item[1], weight= item[0])            
        return G

    @staticmethod
    def show_graph(graph, clusters, showLegend = False, showEdges = False, showWeights = False):
    
        cmap = matplotlib.colormaps.get_cmap('tab20')

        color_map = {}
        color_index = 0

        # Tạo danh sách chú thích (legend)
        legend_items = []

        for cluster_id, nodes in enumerate(clusters):
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
        node_colors = [color_map.get(node, 'grey') for node in graph.nodes()]

        # Vẽ đồ thị
        pos = nx.spring_layout(graph)  
        
        if showEdges:
            nx.draw(graph, pos, node_color=node_colors, node_size=50, edge_color='grey', width=0.5)
        else:
            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=50)
        
        # Thêm chú thích
        if showLegend:
            plt.legend(handles=legend_items, loc="upper right", title="Cluster")

        # Save the plot as an image file
        plt.savefig("graph.png")
        print("Saved the graph as graph.png")


    def remove_singleton(self, graph):
        connected_components = list(nx.connected_components(graph))
    
        # Iterate over each connected component
        for component in connected_components:
            # If the component has only one node, remove it from the graph
            if len(component) == 1:
                graph.remove_node(next(iter(component)))  # Get the single node and remove it
        
        return graph