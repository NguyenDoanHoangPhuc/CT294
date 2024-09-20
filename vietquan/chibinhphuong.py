import pandas as pd
import scipy.stats as stats

# Tạo dữ liệu giả định

header = [
"year","month","day","order","country","session ID",
"page 1 (main category)","page 2 (clothing model)","colour","location","model photography","price","price 2","page"
]
data = pd.read_csv("mapping_data.csv",header=0 )
df = data.head(50000)
# print(df['session ID'])
# print(df['page 2 (clothing model)'])
# Tạo bảng tần suất giữa session ID và colour
meanings =[]
for a in header:
    contingency_table = pd.crosstab(df['session ID'], df[a])
    # Hiển thị bảng tần suất
    # print(contingency_table)

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # In kết quả
    #Giá trị chi-squared cho biết mức độ khác biệt giữa bảng tần suất thực tế và bảng mong đợi.
    #Giá trị chi-squared nhỏ gần với 0 có nghĩa là bảng tần suất thực tế gần giống với bảng mong đợi,
    #ngụ ý rằng hai biến có thể không có mối liên hệ rõ ràng.
    # print(f"Chi-squared: {chi2}")
    #Giá trị P cho biết mức ý nghĩa của kiểm định (thông thường nếu P-value nhỏ hơn 0.05, ta có thể kết luận rằng hai biến có mối liên hệ).
    # print(f"P-value của sessionID so với {a} là : {p}")
    if p>0.05:
        meanings.append(a)
    #Bậc tự do cho biết số thông tin độc lập trong kiểm định.
    # print(f"Degrees of freedom: {dof}")

    # In bảng tần suất mong đợi, thể hiện số lần xuất hiện dự kiến nếu hai biến không có liên hệ.
    # print("Expected frequencies:")
    # print(expected)
    # < 0.05 là có mối quan hệ


print("SessionId so với các thằng còn lại là: ",meanings)












###################################################################################################3





import pandas as pd
import scipy.stats as stats

# Tạo dữ liệu giả định

header = [
"year","month","day","order","country",
"page 1 (main category)","session ID","colour","location","model photography","price","price 2","page"
]
data = pd.read_csv("mapping_data.csv",header=0 )
df = data.head(50000)
# print(df['session ID'])
# print(df['page 2 (clothing model)'])
# Tạo bảng tần suất giữa session ID và colour
meanings =[]
for a in header:
    contingency_table = pd.crosstab(df['page 2 (clothing model)'], df[a])
    # Hiển thị bảng tần suất
    # print(contingency_table)

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # In kết quả
    #Giá trị chi-squared cho biết mức độ khác biệt giữa bảng tần suất thực tế và bảng mong đợi.
    #Giá trị chi-squared nhỏ gần với 0 có nghĩa là bảng tần suất thực tế gần giống với bảng mong đợi,
    #ngụ ý rằng hai biến có thể không có mối liên hệ rõ ràng.
    # print(f"Chi-squared: {chi2}")
    #Giá trị P cho biết mức ý nghĩa của kiểm định (thông thường nếu P-value nhỏ hơn 0.05, ta có thể kết luận rằng hai biến có mối liên hệ).
    # print(f"P-value của page 2 so với {a} là : {p}")
    if p>0.05:
        meanings.append(a)
    #Bậc tự do cho biết số thông tin độc lập trong kiểm định.
    # print(f"Degrees of freedom: {dof}")

    # In bảng tần suất mong đợi, thể hiện số lần xuất hiện dự kiến nếu hai biến không có liên hệ.
    # print("Expected frequencies:")
    # print(expected)
    # < 0.05 là có mối quan hệ


print("Page 2 clothing so với các thằng còn lại là: ",meanings)
