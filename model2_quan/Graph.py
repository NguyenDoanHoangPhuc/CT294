import matplotlib
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

Max_edges_of_node = 40 # Định nghĩa giới hạn số cạnh tối đa của một nút

def add(ArrayLimitEdges, value):
    # Hàm để thêm giá trị vào danh sách cạnh với giới hạn tối đa

    # Nếu danh sách rỗng, thêm giá trị mới vào
    if len(ArrayLimitEdges) == 0:
        ArrayLimitEdges.append(value)
        return
    
    average_distance = value[0]  # Lấy giá trị khoảng cách trung bình của cạnh mới
    min_average_distance_edge = ArrayLimitEdges[0][0]  # Lấy giá trị cạnh nhỏ nhất hiện có

    # Xóa cạnh nhỏ nhất ra nếu cạnh mới thêm vào có giá trị tốt hơn (lớn hơn)
    if len(ArrayLimitEdges) >= Max_edges_of_node:
        if average_distance > min_average_distance_edge:
            ArrayLimitEdges.pop(0)  # Xóa cạnh nhỏ nhất
        else:
            return  # Nếu giá trị mới không tốt hơn, không làm gì cả
    
    # Thêm giá trị mới vào danh sách
    ArrayLimitEdges.append(value)
    # Sắp xếp danh sách theo giá trị tăng dần
    ArrayLimitEdges.sort()  # Sắp xếp lại danh sách sau khi thêm


# Hàm thêm cạnh vào đồ thị
def create_graph(main_grouped,main_graph, attributes, threshold):
        
        G = main_graph.copy()
        df = pd.read_csv('matrix.csv', header=None, skiprows=1)
        # Duyệt qua tính khoảng cách trung bình, tính average distance nếu < ngưỡng threshold thì thêm vào đồ thị
        for idx1, row1 in main_grouped.iterrows():
            session_id1 = row1['session ID']
            limited_arr = []
            for idx2, row2 in main_grouped.iterrows():
                session_id2 = row2['session ID']
                if (session_id2 > session_id1):
                    sum = df.iloc[session_id1- 1, session_id2 - 1]
                    average_distance = sum / len(attributes)
                    if average_distance > threshold:
                        add(limited_arr,[average_distance, session_id2])
                        
            for item in limited_arr:
                G.add_edge(session_id1, item[1], weight = item[0])   
        return G


# Xóa đi những đỉnh đơn lẻ trong đồ thị bằng đồ thị liên thông
def remove_singleton(graph):
        connected_components = list(nx.connected_components(graph))
        for component in connected_components:
            if len(component) == 1:
                graph.remove_node(next(iter(component)))  
        return graph



def show_graph(clusters, G, id, showLegend = True, showEdges = True):
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
                color_map[original_node] = 'blue'
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
    if showEdges:
        nx.draw(G, pos, node_color=node_colors, node_size=10)
    else:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=10)
    # Thêm chú thích
    plt.legend(handles=legend_items, loc="upper right", title="Cluster")
    plt.title(f'Graph {id}')
    plt.show()