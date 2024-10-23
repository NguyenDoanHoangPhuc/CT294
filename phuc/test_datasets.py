from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as stats

# Tạo dữ liệu giả định

all_atributes = [
"day-month",
"country",
"page 1 (main category)","colour","location","model photography","price 2","page",
]
data = pd.read_csv('e-shop clothing 2008.csv', sep = ';')
df = data.copy()

df.loc[:, "day-month"] = df['day'].astype(str) + df['month'].astype(str)
checking_attributes = 'page 2 (clothing model)'


p_values =[]


def calculate_p_value(df, a):
    contingency_table = pd.crosstab(df['page 2 (clothing model)'], df[a])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return p

def calculate_chi2_value(df, a):
    contingency_table = pd.crosstab(df['page 2 (clothing model)'], df[a])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2

def calculate_gini(df, attribute):
    counts = df[attribute].value_counts()
    probabilities = counts / len(df)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

p_values = Parallel(n_jobs=-1)(delayed(calculate_p_value)(df, a) for a in all_atributes)
chi2 = Parallel(n_jobs=-1)(delayed(calculate_chi2_value)(df, a) for a in all_atributes)
ginis = Parallel(n_jobs=-1)(delayed(calculate_gini)(df, a) for a in all_atributes)

p_values_df = pd.DataFrame(columns=all_atributes)
p_values_df.loc[0] = chi2
p_values_df.loc[1] = p_values
p_values_df.loc[2] = ginis
p_values_df.to_csv('p_values.csv', index=False)

# Sort the attributes by the gini value (row index 2) and print the sorted DataFrame
sorted_p_values_df = p_values_df.T.sort_values(by=0, ascending=False)
print(sorted_p_values_df)











