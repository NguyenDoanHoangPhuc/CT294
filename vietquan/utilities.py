
import numpy as np
from nltk import ngrams
from joblib import Parallel, delayed



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




def create_matrix(polar_distance, n_grams, table_data, attributes):

    # Tạo ra ma trận với số lượng bằng với tên của node lớn nhất
    num_nodes = table_data['session ID'].max()
    
    # Tạo từ điển mặc định cho các thuộc tính
    def create_default_dict():
        return {attribute: 0 for attribute in attributes}

    # Tạo ma trận với các từ điển mặc định
    matrix = [[create_default_dict() for _ in range(num_nodes)] for _ in range(num_nodes)]

    def compute_edge(idx1, row1):
        session_id1 = row1['session ID']
        local_edges = []
        for idx2, row2 in table_data.iterrows():
            session_id2 = row2['session ID']
            if session_id2 > session_id1:
                edge_dict = {}
                for attribute in attributes:
                    clothing_models1 = tuple(row1[attribute])
                    clothing_models2 = tuple(row2[attribute])
                    edge_dict[attribute] = 0.5 - polar_distance(clothing_models1, clothing_models2, n_grams)
                local_edges.append((session_id1, session_id2, edge_dict))
        return local_edges

    # Gọi Parallel để chạy song song
    results = Parallel(n_jobs=-1)(
        delayed(compute_edge)(idx1, row1) for idx1, row1 in table_data.iterrows()
    )

    # Cập nhật ma trận từ kết quả
    for local_edges in results:
        for session_id1, session_id2, edge_dict in local_edges:
            matrix[session_id1 - 1][session_id2 - 1] = edge_dict
            matrix[session_id2 - 1][session_id1 - 1] = edge_dict

    return matrix

