import heapq
from joblib import Parallel, delayed
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import numpy as np
from nltk import ngrams
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
    
    return polar_distance_result

# Lớp LimitedSortedArray để lưu trữ top-K phần tử lớn nhất
# Khi thêm một phần tử mới, nếu số lượng phần tử trong mảng nhỏ hơn K, thêm phần tử vào mảng
# Ngược lại, thêm phần tử vào mảng và loại bỏ phần tử nhỏ nhất
class LimitedSortedArray:
    # Hàm khởi tạo với tham số K
    def __init__(self, k):
        # K là giá trị tối đa của mảng, data là mảng lưu trữ dữ liệu
        self.k = k
        self.data = []

    # Hàm thêm phần tử vào mảng
    def add(self, item):
        # Nếu số lượng phần tử trong mảng nhỏ hơn K
        # Thêm phần tử vào mảng, sử dụng heapq để sắp xếp mảng
        if len(self.data) < self.k:
            heapq.heappush(self.data, item)
        else:
            # Nếu số lượng phần tử trong mảng lớn hơn k
            # Nếu nó lớn hơn phần tử nhỏ nhất trong mảng, thêm phần tử vào mảng, xóa phần tử nhỏ nhất đi
            if item > self.data[0]:
                heapq.heappop(self.data)
                heapq.heappush(self.data, item)


# Lớp đồ thị tương đồng
class similarity_graph:
    # Hàm khởi tạo
    def __init__(self, table_data, n_grams):
        # Khởi tạo các biến
        self.n_grams = n_grams
        # Lưu trữ dữ liệu bảng
        self.table_data = table_data
        # Số lượng node
        self.num_nodes = self.get_num_nodes()
        # Đồ thị
        self.graph = nx.Graph()
        
    # Hàm lấy số lượng node
    def get_num_nodes(self):
        return self.table_data['session ID'].max()

    # Hàm viết đồ thị lên một file .dat
    def write_adjacency_matrix(self):
        # Số lượng node
        n = self.num_nodes

        # Lấy dữ liệu từ bảng  
        table_data = self.table_data
        
        # Lấy các thuộc tính từ bảng (bỏ qua sessionID và page 2)
        attributes = table_data.columns[2:]

        # Tạo mảng adjacency matrix để lưu trữ kết quả lên file .dat
        adj_matrix = np.memmap(r'D:\File Code\CT294_Project\CT294\model1\data\adj_matrix.dat', dtype='float32', mode='w+', shape=(n, n))


        # Hàm tính cạnh giữa hai node
        def compute_edge(idx1, row1):
            session_id1 = row1['session ID']
            local_edges = []
            for idx2 in range(idx1 + 1, len(table_data)):
                row2 = table_data.iloc[idx2]
                session_id2 = row2['session ID']
                avg = 0
                for attribute in attributes:
                    clothing_models1 = tuple(row1[attribute])
                    clothing_models2 = tuple(row2[attribute])
                    avg += 0.5 - polar_distance(clothing_models1, clothing_models2)
                avg /= len(attributes)
                local_edges.append((session_id1, session_id2, avg))
            return local_edges

        print("Writing data...")
        # Sử dụng Parallel để tính toán cạnh giữa các node
        results = Parallel(n_jobs=-1)(
            delayed(compute_edge)(idx1, row1) for idx1, row1 in table_data.iterrows()
        )

        print("Updating adjacency matrix...")
        # Cập nhật ma trận từ kết quả
        for local_edges in results:
            for session_id1, session_id2, value in local_edges:
                adj_matrix[session_id1 - 1, session_id2 - 1] = value
        
        print("Finished writing data!")
    
    # Hàm in ma trận kề
    def print_adjacency_matrix(self):
        n = self.num_nodes

        # Tải ma trận kề vào bộ nhớ
        adj_matrix = np.memmap('adj_matrix.dat', dtype='float32', mode='r', shape=(n, n))
        
        # In ma trận kề
        for i in range(n):
            for j in range(n):
                print(f"{adj_matrix[i, j]:.2f}", end=" ")
            print()

    # Hàm tạo đồ thị mẫu
    def create_sample_graph(self,data):
        # Tạo node cho mỗi session ID
        for sessionID in data['session ID']:
            self.graph.add_node(sessionID)

    # Hàm tạo đồ thị k-farthest neighbor (FNG)     
    def get_kfn_graph(self, threshold, k):
        
        # Tải ma trận kề vào bộ nhớ
        adj_matrix = np.memmap(r'D:\File Code\CT294_Project\CT294\model1\data\adj_matrix.dat', dtype='float32', mode='c', shape=(self.num_nodes, self.num_nodes))

        # Tạo một bản sao của đồ thị mẫu
        G = self.graph.copy()

        # Duyệt qua mỗi hàng trong bảng dữ liệu
        for idx1, row1 in self.table_data.iterrows():
            # Lấy session ID
            session_id1 = row1['session ID'] - 1  

            # Lấy khoảng cách từ session_id1 đến các session khác
            distances = adj_matrix[session_id1]
            # Đặt khoảng cách từ session_id1 đến chính nó là âm vô cùng, để tìm cạnh lớn hơn nó
            distances[session_id1] = -np.inf  

            # Lấy k phần tử lớn nhất, trả về chỉ số của các phần tử
            # np.argpartition sắp xếp mảng sao cho các phần tử lớn nhất nằm ở cuối mảng
            # Sau đó lấy k phần tử cuối cùng bằng [-k:]
            top_k_indices = np.argpartition(distances, -k)[-k:]

            # Duyệt qua các chỉ số của các phần tử lớn nhất
            for idx2 in top_k_indices:
                if distances[idx2] > threshold:
                    G.add_edge(session_id1 + 1, idx2 + 1, weight=distances[idx2])

        return G

    # Hàm loại bỏ những bộ phận không liên thông
    def remove_singleton(self, graph):
        # Lấy các thành phần liên thông
        connected_components = list(nx.connected_components(graph))
    
        # Duyệt qua các thành phần liên thông
        for component in connected_components:
            # Nếu thành phần chỉ có 1 node, xóa node đó
            if len(component) == 1:
                graph.remove_node(next(iter(component))) 
        
        return graph

    @staticmethod
    # Hàm lưu đồ thị vào file png
    def save_graph(graph, clusters, showLegend = False, showEdges = False, showWeights = False, idx = 0):
        # Tạo bảng màu
        cmap = matplotlib.colormaps.get_cmap('tab20')

        # Tạo một bảng màu để gán màu cho các node
        color_map = {}
        color_index = 0

        # Tạo danh sách chú thích (legend)
        legend_items = []

        # Duyệt qua các cụm
        for cluster_id, nodes in enumerate(clusters):
            if len(nodes) == 1:
                # Cụm chỉ có 1 phần tử -> gán màu trắng
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

        # Tạo layout cho đồ thị
        pos = nx.spring_layout(graph)  
        
        # Vẽ đồ thị
        if showEdges:
            nx.draw(graph, pos, node_color=node_colors, node_size=50, edge_color='grey', width=0.5)
        else:
            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=50)
        
        # Thêm chú thích
        if showLegend:
            plt.legend(handles=legend_items, loc="upper right", title="Cluster")

        # Lưu đồ thị vào file png, thông báo đã lưu
        plt.savefig(r"D:\File Code\CT294_Project\CT294\model1\data\graph" + str(idx) + ".png")
        print("Saved the graph as graph.png")

    