import time  # Nhập thư viện time để theo dõi thời gian thực thi mã
import leidenalg  # Nhập thư viện leidenalg để thực hiện phân cụm bằng thuật toán Leiden
import pandas as pd  # Nhập pandas để làm việc với dữ liệu dạng bảng (DataFrame)
import igraph as ig  # Nhập igraph để xử lý và phân tích đồ thị
import networkx as nx  # Nhập networkx để xây dựng và thao tác với đồ thị
from nltk import ngrams  # Nhập ngrams từ nltk để tạo n-grams từ văn bản
from Graph import show_graph  # Nhập hàm show_graph từ module Graph để hiển thị đồ thị
from Graph import create_graph  # Nhập hàm create_graph từ module Graph để tạo đồ thị từ dữ liệu
from collections import Counter  # Nhập Counter từ collections để đếm tần suất các đối tượng
from Graph import remove_singleton  # Nhập hàm remove_singleton từ module Graph để loại bỏ các nút đơn trong đồ thị
from utilities import create_matrix  # Nhập hàm create_matrix từ module utilities để tạo ma trận khoảng cách
from utilities import polar_distance  # Nhập hàm polar_distance từ module utilities để tính khoảng cách giữa các n-grams
from utilities import turn_to_cluster  # Nhập hàm turn_to_cluster từ module utilities để chuyển đổi các cụm thành dạng từ điển
from joblib import Parallel, delayed  # Nhập joblib để thực hiện xử lý song song (multi-threading)
import numpy as np
from sklearn.metrics import silhouette_score


start_time = time.time()  # Lưu thời gian bắt đầu thực thi mã

n_grams = 3  # Thiết lập kích thước n-grams là 5 (nghĩa là tạo ra các nhóm từ 5 phần tử liên tiếp)

# Đọc dữ liệu từ file CSV, sử dụng dấu ';' làm dấu phân cách
df = pd.read_csv('e-shop clothing 2008.csv', sep=';')
dataFrame = df.head(n=3000)

# Các thuộc tính mặc định để phân tích
default_attributes = ['colour', 'page 1 (main category)', 'page 2 (clothing model)', 'location', 'page']

# Tên các cột chính trong DataFrame mới
main_columns_name = ['session ID'] + default_attributes  

# Tạo một DataFrame mới để lưu trữ kết quả phân nhóm
main_grouped = pd.DataFrame(columns=main_columns_name)

# Vòng lặp để xử lý từng thuộc tính trong danh sách default_attributes
for idx, attribute in enumerate(default_attributes):
    # Nhóm dữ liệu theo 'session ID' và lấy danh sách các giá trị của thuộc tính hiện tại
    temp = dataFrame.groupby('session ID')[attribute].apply(list).reset_index()
    
    # Lọc các nhóm có độ dài lớn hơn hoặc bằng n_grams (5)
    filtered_temp = temp[temp[attribute].apply(lambda x: len(x) >= n_grams)]
    
    # Tạo n-grams từ các thuộc tính đã lọc và gán cho cột tương ứng trong main_grouped
    main_grouped[attribute] = filtered_temp[attribute].apply(lambda x: list(ngrams(x, n_grams)))
    
    # Gán 'session ID' từ filtered_temp vào main_grouped
    main_grouped['session ID'] = filtered_temp['session ID']



# Đếm số lượng session ID trong DataFrame main_grouped
number_of_sessionID = main_grouped.shape[0]

# Tính số lượng session ID cho tập huấn luyện (90% tổng số session ID)
number_of_train = int(number_of_sessionID * 0.9)

# Tính số lượng session ID cho tập kiểm tra (10% tổng số session ID)
number_of_test = number_of_sessionID - number_of_train

# Lấy 90% đầu tiên của main_grouped làm tập huấn luyện
train_grouped = main_grouped.head(n=number_of_train)

# Lấy 10% cuối cùng của main_grouped làm tập kiểm tra
test_grouped = main_grouped.tail(n=number_of_test)

# Khởi tạo ma trận khoảng cách giữa các session từ tập huấn luyện
create_matrix(train_grouped, default_attributes)

# In ra thông báo rằng việc tạo ma trận đã hoàn thành
print("Create matrix done.")





# Khởi tạo một đồ thị trống sử dụng NetworkX
main_graph = nx.Graph()

# Lặp qua từng session ID trong cột 'session ID' của DataFrame dataFrame
for sessionID in dataFrame['session ID']:
    # Thêm một nút mới vào đồ thị với sessionID là tên nút
    main_graph.add_node(sessionID)
    
    # Gán thuộc tính 'name' cho nút bằng chính sessionID
    main_graph.nodes[sessionID]['name'] = sessionID




def clustering(graph):
    # Chuyển đổi đồ thị NetworkX thành đồ thị igraph
    G_igraph = ig.Graph.from_networkx(graph)
    
    # Tìm partition (phân nhóm) trong đồ thị bằng thuật toán Lei    den, sử dụng ModularityVertexPartition
    partitions = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition, weights="weight")
    
    # Tính toán điểm modularity của partition đã tìm thấy
    modularity_score = partitions.modularity
    
    # Lấy nhãn (labels) của các phần tử trong partition
    labels = np.array(partitions.membership)
    
    # Kiểm tra xem có ít nhất 2 cụm (clusters) không
    if len(set(labels)) >= 2:
        # Tính toán chỉ số silhouette để đánh giá chất lượng của các cụm
        silhouette = silhouette_score(nx.to_numpy_array(graph), labels)
    else:
        # Nếu chỉ có 1 cụm, trả về chỉ số silhouette không hợp lệ
        silhouette = -1  # Invalid silhouette score if only 1 cluster exists
    
    return partitions, modularity_score, silhouette


threshold = 0  # Ngưỡng phân cụm ban đầu
best_graph = None  # Đồ thị tốt nhất tìm được
break_flag = False  # Cờ kiểm soát việc thoát khỏi vòng lặp
best_partitions = None  # Phân nhóm tốt nhất tìm được
best_silhouette = -100  # Giá trị chỉ số silhouette tốt nhất khởi tạo là thấp
best_modularity = -100  # Giá trị modularity tốt nhất khởi tạo là thấp
cluster_loop_count = 0  # Số lần lặp qua các cụm
cluster_max_loop_count = 80  # Số lần lặp tối đa cho quá trình phân cụm
modularity_goal = 0.4  # Mục tiêu cho điểm modularity
min_threshold_increase = 0.01  # Giá trị tăng tối thiểu cho ngưỡng
attributes = default_attributes.copy()  # Sao chép danh sách thuộc tính mặc định
best_attributes = None  # Biến lưu các thuộc tính tốt nhất tìm được


while cluster_loop_count < cluster_max_loop_count and not break_flag:  # Lặp qua số lần tối đa hoặc khi cờ dừng được bật
    cluster_loop_count += 1  # Tăng số lần lặp
    threshold = round(threshold, 2)  # Làm tròn ngưỡng phân cụm đến 2 chữ số thập phân

    graph = remove_singleton(create_graph(train_grouped, main_graph, attributes, threshold))  # Tạo đồ thị và loại bỏ các đỉnh đơn lẻ
    num_edges = graph.number_of_edges()  # Đếm số cạnh trong đồ thị
    if num_edges == 0:  # Nếu không có cạnh nào trong đồ thị
        break_flag = True  # Đặt cờ dừng để thoát khỏi vòng lặp
        break  # Thoát khỏi vòng lặp

    partitions, modularity_score, silhouette = clustering(graph)  # Thực hiện phân cụm trên đồ thị

    if modularity_score > best_silhouette:  # Nếu điểm modularity hiện tại tốt hơn điểm tốt nhất
        best_silhouette = silhouette 
        best_modularity = modularity_score 
        best_partitions = partitions  
        best_graph = graph  
        best_attributes = attributes 

    # Nếu chỉ số modularity đạt yêu cầu thì bật cờ để dừng vòng lặp
    if modularity_score >= modularity_goal:  
        break_flag = True  # Đặt cờ dừng để thoát khỏi vòng lặp
    else:
            threshold += min_threshold_increase  # Tăng ngưỡng phân cụm
            attributes = default_attributes.copy()  # Sao chép lại thuộc tính mặc định để tiếp tục thử nghiệm

    # In thông tin về số lần lặp, điểm modularity tốt nhất, ngưỡng hiện tại và số lượng thuộc tính
    print("Lập:", cluster_loop_count, "Modul:", round(best_silhouette, 7), "Thresh:", round(threshold, 2), "Số thuộc tính:", len(attributes))  


print(best_partitions)

def find_cluster(sessionID1, test_grouped):  # Định nghĩa hàm để tìm cụm cho sessionID nhất định
    min_distance = 100  # Khởi tạo khoảng cách tối thiểu
    cluster_new_node = None  # Khởi tạo biến để lưu ID cụm mới
    clusters = turn_to_cluster(best_partitions)  # Chuyển đổi phân nhóm thành cụm
    
    for cluster_id, cluster in clusters.items():  # Lặp qua từng cụm
        table = train_grouped[train_grouped['session ID'].isin(cluster)]  # Lấy bảng dữ liệu cho cụm hiện tại
        sum = 0  # Khởi tạo tổng khoảng cách cho cụm hiện tại
        
        for attribute in best_attributes:  # Lặp qua các thuộc tính tốt nhất
            all_ngrams = []  # Khởi tạo danh sách để chứa tất cả n-grams
            
            # Lặp qua từng hàng trong cột đã chỉ định
            for row in table[attribute]:  
                all_ngrams.extend(row)  # Mở rộng danh sách với n-grams từ hàng hiện tại
            
            # Đếm tần suất của từng n-gram
            ngram_counts = Counter(all_ngrams)  
            ngrams_only = [ngram for ngram, count in ngram_counts.most_common(1)]  # Lấy n-gram phổ biến nhất

            clothing_models1 = test_grouped[test_grouped['session ID'] == sessionID1][attribute].values[0]  # Lấy mô hình quần áo từ test_grouped
            clothing_models2 = list(ngrams_only[0])  # Chuyển đổi n-gram phổ biến thành danh sách
            if len(clothing_models2) == 0:  # Nếu không có n-gram nào
                continue 
            
            sum += polar_distance(clothing_models1, list(ngrams(clothing_models2, 2)))  # Tính khoảng cách và cộng vào tổng
            
        if sum < min_distance:  # Nếu tổng khoảng cách nhỏ hơn khoảng cách tối thiểu
            min_distance = sum  # Cập nhật khoảng cách tối thiểu
            cluster_new_node = cluster_id  # Cập nhật ID cụm mới
            
    return cluster_new_node  # Trả về ID cụm gần nhất




total_f1_score = 0  # Khởi tạo biến tổng f1 score
total_pre_score = 0  # Khởi tạo biến tổng precision score
total_recall_score = 0  # Khởi tạo biến tổng recall score
total_count = 0  # Khởi tạo biến đếm tổng

def test_model(sessionID, best_attributes=best_attributes, best_partitions=best_partitions, data=dataFrame, test_grouped=test_grouped, train_grouped=train_grouped):  # Định nghĩa hàm kiểm tra mô hình
    clusters = turn_to_cluster(best_partitions)  # Chuyển đổi phân nhóm thành cụm
    check_attributes = best_attributes.copy()  # Sao chép thuộc tính tốt nhất

    if 'page 2 (clothing model)' in check_attributes:  # Nếu thuộc tính 'page 2 (clothing model)' có trong danh sách
        check_attributes.remove('page 2 (clothing model)')  # Xóa thuộc tính này khỏi danh sách
    clusterID = find_cluster(sessionID, test_grouped)  # Tìm cụm cho sessionID hiện tại
    prediction_product = []  # Khởi tạo danh sách để lưu sản phẩm dự đoán

    # Danh sách sản phẩm thực tế của người dùng
    product_list = list(data[data['session ID'] == sessionID]['page 2 (clothing model)'])  # Lấy danh sách sản phẩm thực tế cho sessionID
    one_third = len(product_list) // 3  # Chia danh sách sản phẩm thành ba phần
    given_product_list = product_list[:one_third]  # Lấy một phần sản phẩm để đưa vào

    true_product_list = product_list[one_third:]  # Lấy danh sách sản phẩm thực tế còn lại

    for given_product in given_product_list:  # Duyệt qua từng sản phẩm đã cho
        # Duyệt qua các node trong cùng một cụm
        for node in clusters[clusterID]:  # Lặp qua các node trong cụm hiện tại
            # Lấy ra danh sách sản phẩm của các node trong cụm
            product_id_list = train_grouped[train_grouped['session ID'] == node]['page 2 (clothing model)'].values[0]  # Lấy danh sách sản phẩm của node
            
            for product_id in product_id_list:  # Lặp qua danh sách sản phẩm
                # Nếu phần tử đầu tiên của n-gram giống với phần tử đầu tiên của danh sách được cho trước thì thêm vào dự đoán
                if product_id[0] == given_product:  
                    if len(set(prediction_product)) < len(true_product_list) * 3.5:  # Kiểm tra nếu số lượng dự đoán chưa đủ
                        prediction_product.extend(product_id[0:])  # Thêm sản phẩm vào danh sách dự đoán

    tanxuat = Counter(prediction_product)  # Đếm tần suất các sản phẩm trong danh sách dự đoán
    matching_elements = set(prediction_product) & set(true_product_list)  # Tìm các sản phẩm khớp giữa dự đoán và thực tế
    matching_count = len(matching_elements)  # Đếm số lượng sản phẩm khớp

    # Tính precision, recall và f1 score
    if len(prediction_product) != 0: 
        precision = matching_count / (len(prediction_product))  
    else:
        precision = 0  
    
    if matching_count != 0:  
        recall = matching_count / (len(true_product_list))  
    else:
        recall = 0

    if precision + recall == 0:  
        f1_score = 0  
    else:
        f1_score = 2 * precision * recall / (precision + recall) 
    return (precision, recall, f1_score) 



results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(test_model)(sessionID) for sessionID in list(test_grouped['session ID'])
)

for precision, recall, f1_score in results:
    total_f1_score += f1_score
    total_pre_score += precision
    total_recall_score += recall
    total_count += 1
print("Giá trị silhouette tốt nhất: ", best_silhouette) 
print("Giá trị modularity tốt nhất: ", best_modularity) 
print("Điểm F1 tổng cộng: ", total_f1_score / total_count)  
print("Điểm precision tổng cộng: ", total_pre_score / total_count)
print("Điểm recall tổng cộng: ", total_recall_score / total_count)  


end_time = time.time()
print(f"Thời gian thực thi: {end_time - start_time :.6f} giây") 
show_graph(best_partitions, best_graph, f"Big Graph ", showEdges=False)




