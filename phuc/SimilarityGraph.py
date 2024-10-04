import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# Lớp thể hiện đồ thị tương đồng
# Đồ thị tương đồng gồm có: Đồ thị nx, ngưỡng tương đồng, dữ liệu từ bảng, các thuộc tính
class SimilarityGraph:
    def __init__(self, data, threshold, attributes, distance_function):
        self.data = data
        self.graph = self.build_graph(distance_function=distance_function, attributes=attributes, threshold=threshold)

    def build_graph(self, distance_function, attributes, threshold):
        G = nx.Graph()

        for sessionID in self.data['session ID']:
            G.add_node(sessionID)

        for idx1, row1 in self.data.iterrows():
            session_id1 = row1['session ID']

            for idx2, row2 in self.data.iterrows():
                session_id2 = row2['session ID']
                if (session_id2 <= session_id1):
                    continue
                sum = 0
                for attribute in attributes:
                    clothing_models1 = tuple(row1[attribute])
                    clothing_models2 = tuple(row2[attribute])
                    sum += 0.5 - distance_function(clothing_models1, clothing_models2, 2)
                average_distance = sum / len(attributes)
                if average_distance > threshold:
                    G.add_edge(session_id1, session_id2, weight= average_distance)                
        return G

    def show_graph(self, clusters, showLegend = False, showEdges = False, showWeights = False):
        
        new_graph = self.graph.copy()

        cmap = matplotlib.colormaps.get_cmap('tab20')

        color_map = {}
        color_index = 0

        # Tạo danh sách chú thích (legend)
        legend_items = []

        for cluster_id, nodes in enumerate(clusters):
            if len(nodes) == 1:
                # Cụm chỉ có 1 phần tử -> gán màu xanh dương
                for node in nodes:
                    color_map[node] = 'blue'
            else:
                # Cụm có nhiều phần tử -> gán màu từ bảng màu
                current_color = cmap(color_index)
                for node in nodes:
                    color_map[node] = current_color
                color_index += 1
                legend_items.append(mpatches.Patch(color=current_color, label=f"Cluster {cluster_id}"))  # Chú thích cho cụm có nhiều phần tử

        # Lấy danh sách màu cho các node trong đồ thị
        node_colors = [color_map.get(node, 'grey') for node in new_graph.nodes()]

        # Vẽ đồ thị
        pos = nx.spring_layout(new_graph)  
        
        if showEdges:
            # Hiển thị trọng số của các cạnh (trọng số là nhãn cạnh)
            # edge_labels = nx.get_edge_attributes(new_graph, 'weight')  # Lấy trọng số các cạnh
            # nx.draw_networkx_edge_labels(new_graph, pos, edge_labels=edge_labels)  # Vẽ nhãn cho các cạnh
            nx.draw(new_graph, pos, node_color=node_colors, node_size=50, edge_color='grey', width=0.5, with_labels=showWeights)
        else:
            nx.draw_networkx_nodes(new_graph, pos, node_color=node_colors, node_size=50)
        
        # Thêm chú thích
        if showLegend:
            plt.legend(handles=legend_items, loc="upper right", title="Cluster")

        plt.show()

    def get_connected_components(self):
        nx.connected_components(self.graph)
        # # Tìm tất cả các thành phần liên thông
        connected_components = list(nx.connected_components(self.graph))

        # # Lọc ra những bộ phận liên thông có nhiều hơn 2 đỉnh
        filtered_components = [component for component in connected_components if len(component) > 2]

        # # Tạo các đồ thị con từ mỗi thành phần liên thông
        sub_graphs = [self.graph.subgraph(component).copy() for component in filtered_components]
        return sub_graphs