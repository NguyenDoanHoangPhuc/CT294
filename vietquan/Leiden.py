import time
import leidenalg
import pandas as pd
import igraph as ig
import networkx as nx
from Graph import show_graph
from Graph import create_graph
from Graph import remove_singleton
from utilities import create_matrix
from utilities import polar_distance
from scipy.stats import chi2_contingency

start_time = time.time()

# n_grams = 5 # threshold = 0.09 - 30000
n_grams = 4 # threshold = 0 - 30000

# Đọc dữ liệu từ CSV
df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')
# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
dataFrame = df.head(n = 1000)





default_attributes = ['page 2 (clothing model)', 'colour', 'location', 'model photography', 'price 2', 'page']
main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)


#Nhóm các sessionId với các trường khác lại thành 1 dòng sessionId
for idx, attribute in enumerate(default_attributes):
    temp = dataFrame.groupby('session ID')[attribute].apply(list).reset_index()
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) >= n_grams)]
    main_grouped[attribute] = filtered_temp[attribute]
    main_grouped['session ID'] = filtered_temp['session ID']

# Khởi tạo ma trận khoảng cách giữa các session
matrix = create_matrix( polar_distance, n_grams, main_grouped, default_attributes )
print("Create matrix done.")





# Vì tất cả đồ thị đều giống nhau về số lượng node chỉ khác nhau về cạnh
# Nên chỉ cần khởi tạo node 1 lần, khi duyệt qua enum nó mất đi tên node nên phải gắn name cho nó 

main_graph = nx.Graph()
for sessionID in dataFrame['session ID']:
   main_graph.add_node(sessionID)
   main_graph.nodes[sessionID]['name'] = sessionID # Chỉ thuật toán của tui





# Hàm này xóa thuộc tính nếu như chi-square > 0.05
def remove_attribute(attributes, graph):
    sessionID_list = list(graph.nodes())

    filtered_df = dataFrame[dataFrame['session ID'].isin(sessionID_list)]
    filtered_columns_df = filtered_df.loc[:, 'page 1 (main category)': 'page']
                
    G_igraph = ig.Graph.from_networkx(graph)
    session_to_cluster = {}

    # Xử lí thêm trường cluster có giá trị là cụm mà nó thuộc về
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




# Tiến hành phân cụm, trả về cụm và chỉ số modul
def clustering(graph):
    G_igraph = ig.Graph.from_networkx(graph)
    partitions = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition,weights="weight")
    modularity_score = partitions.modularity
    return partitions, modularity_score




###################################################################################################################

# Khởi tạo giá trị ban đầu
threshold = 0
best_graph = None
break_flag = False
best_partitions = None
best_silhouette = -100
cluster_loop_count = 0
cluster_max_loop_count = 80
modularity_goal = 0.5
min_threshold_increase = 0.01
attributes = default_attributes.copy()


while cluster_loop_count < cluster_max_loop_count and not break_flag:
    cluster_loop_count += 1
    threshold = round(threshold, 2)

    graph =  remove_singleton(create_graph(main_grouped, main_graph, matrix, attributes, threshold))

    num_edges = graph.number_of_edges()
    if num_edges == 0:
        break_flag = True
        break

    partitions, modularity = clustering(graph)

    if modularity > best_silhouette:
       best_silhouette = modularity
       best_partitions = partitions
       best_graph = graph

    # Nếu chỉ số phù hợp thì bật cờ để dừng vòng lập
    if modularity >= modularity_goal:
        break_flag = True

    # Nếu xóa được thuộc tính => lập lại | không xóa được thuộc tính tăng ngưỡng
    else:
        response = remove_attribute(attributes, graph)
        if response[1] == False:
            threshold += min_threshold_increase
            attributes = default_attributes.copy()     
    print("Lập:",cluster_loop_count, "Modul:",round(best_silhouette, 7) ,"Thresh:", round(threshold, 2) , "Số thuộc tính:", len(attributes))    

print(threshold)
print(partitions)


end_time = time.time()
print(f"Thời gian thực thi: {end_time - start_time :.6f} giây") 


show_graph(partitions, best_graph, f"Big Graph ", showEdges=False)
