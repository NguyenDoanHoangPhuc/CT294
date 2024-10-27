import pandas as pd
import numpy as np
import gc
import time 
from nltk import ngrams
from collections import Counter
from joblib import Parallel, delayed

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.graph_objects as go

def read_data_csv():
    '''Đọc file csv tùy chỉnh và xử lý dữ liệu'''

    # Đường dẫn file csv
    file_path = 'e-shop clothing 2008.csv'

    # Cột cần đọc
    use_cols = ['session ID', 'page 1 (main category)', 'page 2 (clothing model)', 'colour', 'location', 'page']
    
    # Ký tự phân cách
    _sep = ';'
    
    # Kiểu dữ liệu vừa đủ lưu trữ cho mỗi cột sử dụng để giảm bộ nhớ sử dụng 
    dtypes = {
        'session ID': np.uint16,
        'page 1 (main category)': np.uint8,
        'page 2 (clothing model)': str,  
        'colour': np.uint8,
        'location': np.uint8,
        'page': np.uint8
    }
    
    # Chọn số hành động để phân tích trong chuỗi hành động
    n_grams = 3
    
    # Đọc file csv với những thông tin ở trên
    data = pd.read_csv(file_path, sep=_sep, dtype=dtypes, usecols=use_cols)
    
    # Hiển thị dữ liệu đọc vào
    print("\nDữ liệu đọc vào:\n", data)

    # Hiển thị dữ liệu trước khi loại bỏ các session ID có số lần click chuột nhỏ hơn n_grams
    print("\nDữ liệu trước khi loại bỏ các session ID có số lần click chuột nhỏ hơn n_grams:\n", data)
    
    # Chỉ giữ lại những session ID có số lần click chuột lớn hơn bằng n_grams
    data = data.groupby('session ID').filter(lambda x: len(x) >= n_grams)
    
    # Hiển thị dữ liệu sau khi loại bỏ các session ID có số lần click chuột nhỏ hơn n_grams
    print("\nDữ liệu sau khi loại bỏ các session ID có số lần click chuột nhỏ hơn n_grams:\n", data)

    # Với mỗi session ID, chuyển các chuỗi giá trị theo từng thuộc tính thành danh sách các n-gram và làm mới chỉ số dòng của dữ liệu
    data_ngrams = data.groupby('session ID').agg(lambda x: list(ngrams(x.to_list(), n_grams))).reset_index()

    # Hiển thị dữ liệu n-grams
    print("\nDữ liệu n-grams:\n", data_ngrams)
    
    # Giải phóng bộ nhớ
    del data
    gc.collect()
    
    # Trả về dữ liệu n-grams đã xử lý
    return data_ngrams

def polar_distance(ngrams_list1, ngrams_list2):
    '''Hàm tính khoảng cách Polar giữa hai chuỗi thứ tự'''
    
    # Đếm số lần xuất hiện của các phần tử trong 2 danh sách n-grams
    counter1, counter2 = Counter(ngrams_list1), Counter(ngrams_list2)
    
    # Tìm phần hợp giữa 2 danh sách n-grams
    union_ngrams = set(counter1.keys()) | set(counter2.keys())
    '''Polar distance = 1/pi x arccos(tích vô hướng 2 vector / (độ dài vectơ 1 * độ dài vectơ 2))'''
    
    # Tạo 2 vector tần suất chuẩn hóa cho 2 danh sách n-grams
    frequency_vector1 = [counter1.get(ngram, 0) / len(ngrams_list1) for ngram in union_ngrams]
    frequency_vector2 = [counter2.get(ngram, 0) / len(ngrams_list2) for ngram in union_ngrams]
    
    # Tính tích vô hướng 2 vectơ
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
    
    # Trả về khoảng cách polar
    return polar_distance_result

def polar_distance_matrix(data_ngrams, attributes):
    '''Hàm tạo ma trận khoảng cách song song'''

    # Số lượng session ID
    number_of_nodes = len(data_ngrams)

    def row_distance(row_index, data_ngrams, attributes):
        '''Hàm tính khoảng cách giữa session ID tại row_index với các session ID còn lại'''

        # Tạo danh sách khoảng cách
        row_matrix = np.empty(number_of_nodes, dtype=np.float16)

        # Đặt khoảng cách từ session ID tại row_index đến chính nó là 0
        row_matrix[row_index] = 0

        # Tính khoảng cách giữa session ID tại row_index với các session ID còn lại
        for col_index in range(row_index + 1, number_of_nodes):
            # Tính tổng khoảng cách giữa session ID tại row_index với các session ID còn lại
            total_distance = sum(
                polar_distance(data_ngrams.at[row_index, attr], data_ngrams.at[col_index, attr])
                for attr in attributes
            )
            # Tính khoảng cách trung bình = tổng khoảng cách / số lượng thuộc tính
            row_matrix[col_index] = total_distance / len(attributes)

        # Trả về danh sách khoảng cách
        return row_matrix
    
    # Tạo ma trận khoảng cách bằng cách sử dụng song song
    distance_matrix = Parallel(n_jobs=-1)( # n_jobs = -1 nghĩa là sử dụng tất cả các CPU có sẵn
        # delayed(row_distance) là hàm trả về khoảng cách giữa session ID tại row_index với các session ID còn lại
        delayed(row_distance)(row_index, data_ngrams, attributes)
        for row_index in range(number_of_nodes)
    )

    # Đối xứng ma trận khoảng cách
    for row_index in range(number_of_nodes):
        for col_index in range(row_index):
            distance_matrix[row_index][col_index] = distance_matrix[col_index][row_index]
    
    # Trả về ma trận khoảng cách song song
    return distance_matrix

def K_means_clustering(data_ngrams, current_attributes, n_clusters):
    '''Hàm thực hiện phân cụm K-Means'''

    # Hiển thị các thuộc tính hiện tại
    print("\nPhân cụm K-Means với những thuộc tính hiện tại:\n", current_attributes)
    
    # Hiển thị số lượng cụm
    print("\nSố lượng cụm: ", n_clusters)
    
    # Tạo ma trận khoảng cách polar giữa các session ID
    distance_matrix = polar_distance_matrix(data_ngrams, current_attributes)

    # Chuyển đổi ma trận khoảng cách thành tọa độ điểm trong không gian 3 chiều bằng MDS
    mds = MDS(
        n_components=3, 
        metric=False, 
        random_state=42, 
        dissimilarity='precomputed',
        normalized_stress=False
    )
    '''n_components = 3 nghĩa là chuyển đổi ma trận khoảng cách thành tọa độ điểm trong không gian 3 chiều
    metric = False nghĩa là sẽ tập trung vào việc bảo toàn thứ tự của các khoảng cách, không phải giá trị tuyệt đối của chúng
    random_state = 42 để đảm bảo kết quả có thể lặp lại
    dissimilarity = 'precomputed' nghĩa là ma trận khoảng cách đã được tính toán trước, không cần MDS tính toán lại
    normalized_stress = False để không sử dụng chỉ số stress để đánh giá chất lượng việc giảm chiều dữ liệu'''
    
    # Chuyển đổi ma trận khoảng cách thành tọa độ điểm
    points = mds.fit_transform(distance_matrix) 

    # Giải phóng bộ nhớ
    del mds
    del distance_matrix
    gc.collect()
    
    # Thực hiện thuật toán K-means
    kmeans = KMeans(n_clusters, random_state=42)
    
    # Thực hiện phân cụm
    clusters = kmeans.fit_predict(points)

    # Vẽ kết quả gom cụm K-Means sử dụng Plotly (tương tác)
    plot_3d_scatter(points, n_clusters, kmeans, clusters)
    
    # Tính chỉ số silhouette
    silhouette_avg = silhouette_score(points, clusters)

    # Hiển thị chỉ số silhouette
    print(f"Chỉ số silhouette : {silhouette_avg}")
    
    # Giải phóng bộ nhớ
    del points
    del kmeans
    gc.collect()
    
    # Trả về kết quả phân cụm và chỉ số silhouette
    return clusters, silhouette_avg

def plot_3d_scatter(points, n_clusters, kmeans, clusters):
    '''Vẽ kết quả gom cụm K-Means sử dụng Plotly (tương tác)'''

    # Vẽ các điểm
    fig = go.Figure(data=[go.Scatter3d(
        # Tọa độ của các điểm
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        # Chế độ vẽ
        mode='markers',
        # Các thuộc tính của điểm vẽ
        marker=dict(
            # Kích thước của điểm
            size=5,
            # Màu sắc của điểm
            color=clusters,
            # Màu sắc của điểm
            colorscale='Viridis',
            # Độ mờ của điểm
            opacity=0.8
        ),
        # Nội dung hiển thị khi hover: Cluster, X, Y, Z
        text=[f'Cluster: {i}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}'for i, x, y, z in zip(clusters, points[:, 0], points[:, 1], points[:, 2])],
        # Chế độ hiển thị khi hover
        hoverinfo='text'
    )])

    # Lấy tọa độ các centroid
    centroids = kmeans.cluster_centers_
    # Vẽ các centroid
    fig.add_trace(go.Scatter3d(
        # Tọa độ của các centroid
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        # Chế độ vẽ
        mode='markers',
        # Các thuộc tính của điểm vẽ
        marker=dict(
            # Màu sắc của điểm
            color='red',
            # Kích thước của điểm
            size=10,
            # Kí hiệu của điểm
            symbol='diamond'
        ),
        # Nội dung hiển thị khi hover: Centroid
        text=[f'Centroid {i+1}' for i in range(n_clusters)],
        # Chế độ hiển thị khi hover
        hoverinfo='text'
    ))
    # Vẽ layout
    fig.update_layout(
        # Tiêu đề
        title=f'K-means Clustering (K={n_clusters})',
        # Ẩn nền hiển thị cho mặt phẳng
        scene=dict(
            xaxis=dict(showbackground=False, title='MDS Feature 1'),
            yaxis=dict(showbackground=False, title='MDS Feature 2'),
            zaxis=dict(showbackground=False, title='MDS Feature 3'),
            aspectmode='cube'
        ),
        # Căn chỉnh kích thước tự động toàn màn hình
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    # Hiển thị đồ thị
    fig.show()

class LimitedSortedArray:
    '''Lớp để lưu trữ danh sách các phần tử với số lượng tối đa và sắp xếp theo thứ tự tăng dần'''
    def __init__(self, max_length):
        '''Hàm khởi tạo danh sách với số lượng phần tử tối đa'''
        self.max_length = max_length
        self.data = list()
    def add(self, value):
        '''Hàm thêm phần tử vào danh sách'''
        # Nếu danh sách chưa đầy
        if len(self.data) < self.max_length:
            # Thêm giá trị mới vào mảng
            self.data.append(value)
            # Sắp xếp mảng theo thứ tự tăng dần
            self.data.sort(key=lambda x: x[0])
        # Nếu danh sách đã đầy
        else:
            # Nếu giá trị mới nhỏ hơn phần tử lớn nhất (cuối mảng)
            if value[0] < self.data[-1][0]:
                # Xóa phần tử cuối
                self.data.pop()
                # Thêm giá trị mới vào mảng
                self.data.append(value)
                # Sắp xếp mảng theo thứ tự tăng dần
                self.data.sort(key=lambda x: x[0])
            # Không thêm nếu giá trị mới nhỏ hơn hoặc bằng phần tử nhỏ nhất
            else:
                return
            
def find_cluster(index1_in_test_data, n_clusters, clusters, test_data, train_data, check_attributes):
    '''Hàm tìm cụm của session ID trong tập test'''

    # Tạo danh sách LimitedSortedArray lưu trữ khoảng cách tới các session ID trong tập train
    list_neighbours = LimitedSortedArray(n_clusters + 1)

    # Tính khoảng cách từ session ID tại index1_in_test_data đến các session ID trong tập train
    for index2_in_train_data in range(len(train_data)):
        # Tính tổng khoảng cách giữa session ID tại index1_in_test_data với các session ID trong tập train
        total_distance = sum(
            polar_distance(test_data.at[index1_in_test_data, check_attribute], train_data.at[index2_in_train_data, check_attribute])
            for check_attribute in check_attributes
        )
        # Tính khoảng cách trung bình = tổng khoảng cách / số lượng thuộc tính
        average_distance = total_distance / len(check_attributes)
        # Thêm khoảng cách và index của session ID trong tập train vào danh sách
        list_neighbours.add((average_distance, index2_in_train_data))
    
    # Lấy các index của session ID trong tập train tương ứng với khoảng cách nhỏ nhất
    neighbors_clusters = [clusters[index2_in_train_data] for _, index2_in_train_data in list_neighbours.data]
   
    most_common_clusterID = Counter(neighbors_clusters).most_common(1)[0][0]

    # Trả về cụm của session ID tại index1_in_test_data
    return most_common_clusterID

def test_model(index1_in_test_data, check_attributes, test_data, train_data, n_clusters, clusters):
    '''Hàm kiểm tra mô hình với tập test'''

    # Tìm cụm của session ID tại index1_in_test_data
    clusterID = find_cluster(index1_in_test_data, n_clusters, clusters, test_data, train_data, check_attributes)
    
    # Tạo danh sách chứa các sản phẩm trong session ID tại index1_in_test_data
    product_list = list()

    # Lặp qua các sản phẩm trong các n-grams của session ID tại index1_in_test_data, nếu sản phẩm không có trong danh sách thì thêm vào danh sách
    for n_grams in test_data.at[index1_in_test_data, 'page 2 (clothing model)']:
        for product in n_grams:
            if product not in product_list:
                product_list.append(product)
    
    # Chia danh sách sản phẩm thành 2 phần: 1/3 sản phẩm được cho trước, 2/3 sản phẩm cần dự đoán
    one_third = int(len(product_list) / 3)
    given_product_list =  product_list[:one_third]
    true_product_list = product_list[one_third:]
    
    # Tạo một Series từ mảng clusters với index tương ứng với train_data
    cluster_series = pd.Series(clusters, index=train_data.index)
    
    # Lấy các session ID thuộc cụm clusterID
    session_ID_of_clusterID = train_data[cluster_series.isin([clusterID])]
    
    # Tạo danh sách chứa các n-grams của các session ID thuộc cụm clusterID
    n_grams_list_of_cluster = [
        n_grams
        for n_grams_list in session_ID_of_clusterID['page 2 (clothing model)'] 
        for n_grams in n_grams_list
    ]

    # Tạo danh sách chứa các sản phẩm dự đoán
    prediction_product = set()

    # Lặp qua các sản phẩm được cho trước
    for given_product in given_product_list:
        count = 0
        for n_grams in n_grams_list_of_cluster:
            for i in range(len(n_grams) - 1):
                if given_product == n_grams[i]:
                    next_product = n_grams[i + 1]
                    if count <= len(true_product_list)*2:
                        prediction_product.add(next_product)
                        count += 1

    # Tìm sản phẩm dự đoán trùng với sản phẩm thực tế
    matching_elements = prediction_product & set(true_product_list)
    
    # Đếm số lượng sản phẩm dự đoán trùng với sản phẩm thực tế
    matching_count = len(matching_elements)

    # Tính các chỉ số f1-score, precision, recall để đánh giá mô hình 
    if len(prediction_product) != 0:
        precision = matching_count / len(prediction_product)
    else:
        precision = 0
    
    if matching_count != 0:
        recall = matching_count / len(true_product_list)
    else:
        recall = 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return f1_score, precision, recall

def Evaluation_n_clusters(n_clusters):
    '''Hàm đánh giá chất lượng phân cụm với số lượng cụm là n_clusters'''
    
    # Các thuộc tính dùng để phân cụm
    attributes = ['page 1 (main category)', 'colour', 'location', 'page']   

    # Hiển thị các thuộc tính dùng để phân cụm
    print("\nCác thuộc tính dùng để phân cụm:\n", attributes)

    # Đọc và xử lý dữ liệu
    data_ngrams = read_data_csv()

    # Chia tập dữ liệu thành tập train và tập test
    _90_percent = int(len(data_ngrams) * 0.9)    
    train_data = data_ngrams.iloc[0:_90_percent]
    test_data = data_ngrams.iloc[_90_percent:]

    # Giải phóng bộ nhớ
    del data_ngrams
    gc.collect()

    # Reset index
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    # Phân cụm K-Means với tập train
    clusters, silhouette_avg = K_means_clustering(train_data, attributes, n_clusters)

    # Kiểm tra mô hình với tập test
    total_f1_score = [
        test_model(index1_in_test_data, attributes, test_data, train_data, n_clusters, clusters)
        for index1_in_test_data in range(len(test_data))
    ]

    # Giải phóng bộ nhớ
    del train_data, test_data
    gc.collect()

    # Tính các chỉ số f1-score, precision, recall trung bình
    average_f1_score = sum(f1_score for f1_score, _, _ in total_f1_score) / len(total_f1_score)
    average_precision = sum(precision for _, precision, _ in total_f1_score) / len(total_f1_score)
    average_recall = sum(recall for _, _, recall in total_f1_score) / len(total_f1_score)

    # Hiển thị các chỉ số f1-score, precision, recall trung bình
    print("\nf1-score:\n", average_f1_score)
    print("\nprecision:\n", average_precision)
    print("\nrecall:\n", average_recall)

    # Trả về số lượng cụm, chỉ số silhouette, các chỉ số f1-score, precision, recall trung bình
    return n_clusters, silhouette_avg, average_f1_score, average_precision, average_recall

if __name__=='__main__':
    # Đánh dấu thời gian bắt đầu
    start_time = time.time()

    # Đánh giá chất lượng phân cụm với số lượng cụm chỉ định 
    Evaluation_n_clusters(6)

    # Đánh dấu thời gian kết thúc
    end_time = time.time()
    
    # Hiển thị thời gian chạy
    print("Time taken: ", end_time - start_time)
    
    
    

    
    
