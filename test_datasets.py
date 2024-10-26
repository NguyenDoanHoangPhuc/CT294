from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as stats

# Những thuộc tính cần kiểm tra
all_atributes = [
"day-month", "country", "page 1 (main category)","colour","location","model photography","price 2","page",
]

# Đọc dữ liệu từ CSV
data = pd.read_csv('e-shop clothing 2008.csv', sep = ';')
df = data.copy()

# Tạo cột day-month từ cột day và month
df.loc[:, "day-month"] = df['day'].astype(str) + df['month'].astype(str)

p_values =[]

# Tính p-value cho từng thuộc tính
def calculate_p_value(df, a):
    contingency_table = pd.crosstab(df['page 2 (clothing model)'], df[a])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return p

# Tính giá trị chi2 cho từng thuộc tính
def calculate_chi2_value(df, a):
    contingency_table = pd.crosstab(df['page 2 (clothing model)'], df[a])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2

# Tính gini cho từng thuộc tính
def calculate_gini(df, attribute):
    counts = df[attribute].value_counts()
    probabilities = counts / len(df)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Tính p-value, chi2 và gini cho tất cả các thuộc tính, sử dụng Parallel để tăng tốc độ tính toán
p_values = Parallel(n_jobs=-1)(delayed(calculate_p_value)(df, a) for a in all_atributes)
chi2 = Parallel(n_jobs=-1)(delayed(calculate_chi2_value)(df, a) for a in all_atributes)
ginis = Parallel(n_jobs=-1)(delayed(calculate_gini)(df, a) for a in all_atributes)

# Tạo DataFrame chứa p-value, chi2 và gini của tất cả các thuộc tính
test_df = pd.DataFrame(columns=all_atributes)
test_df.loc[0] = chi2
test_df.loc[1] = p_values
test_df.loc[2] = ginis

# Lưu DataFrame chứa p-value, chi2 và gini của tất cả các thuộc tính ra file CSV
test_df.to_csv('p_values.csv', index=False)

# Sắp xếp DataFrame theo Chi-Square
sorted_df = test_df.T.sort_values(by=0, ascending=False)
print(sorted_df)











