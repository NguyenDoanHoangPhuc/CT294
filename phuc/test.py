# Đọc dữ liệu từ CSV
import time
import pandas as pd
from nltk import ngrams
import seaborn as sns

df = pd.read_csv('e-shop clothing 2008.csv', sep = ';')

# Lấy ra n dòng đầu tiên từ dữ liệu cho trước
data = df.head(n = 22942)

#34082
default_attributes = ['page 1 (main category)','page 2 (clothing model)','colour', 'location', 'model photography', 'price 2', 'page']

result = []
start_time = time.time()

n_grams = 3
main_columns_name = ['session ID'] + default_attributes 
main_grouped = pd.DataFrame(columns = main_columns_name)


import matplotlib.pyplot as plt

# Set up the matplotlib figure


# # Select one attribute to plot
# attribute_to_plot = 'page 1 (main category)'

# # Plot the distribution of the selected attribute
# sns.histplot(data[attribute_to_plot], kde=True, ax=ax)
# ax.set_title(f'Distribution of {attribute_to_plot}')
# ax.set_xlabel('Value')
# ax.set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()
# Plot the distribution of each default attribute in the same plot
for attribute in default_attributes:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data[attribute], kde=True, ax=ax, label=attribute)
    ax.set_title(f'Distribution of {attribute}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.tight_layout()
    plt.show()



number_of_sessionID = main_grouped.shape[0]
number_of_train = int(number_of_sessionID * 0.9)
number_of_test = number_of_sessionID - number_of_train



train_grouped = main_grouped.head(n = number_of_train)
test_grouped = main_grouped.tail(n = number_of_test)