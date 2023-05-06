import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import math

# Read result
setting_name = 'v2.11d.gpt3.6'
result_path = f'cache/{setting_name}/facts.csv'
df = pd.read_csv(result_path)
df = df.loc[:, ['s', 'o', 'c_original','c_all']]


# Read mindegree dict
# mindegree_dictionary = pickle.load(open(f'mindegree_dictionary.pkl', 'rb'))

# tmp
tmp_path = f'real_degree_dataset_dictionary.pkl'
if os.path.exists(tmp_path):
    real_degree_dictionary = pickle.load(open(tmp_path, 'rb'))
else:
    real_degree_dictionary = pickle.load(open(f'real_degree_dictionary.pkl', 'rb'))
    tmp_dictionary = {}
    for key in df['s'].unique():
        tmp_dictionary[key] = real_degree_dictionary[key]
    for key in df['o'].unique():
        tmp_dictionary[key] = real_degree_dictionary[key]
    real_degree_dictionary = tmp_dictionary
    pickle.dump(real_degree_dictionary, open(tmp_path, 'wb'))


# # Add mindegree column
# df['mindegree_s'] = df.apply(lambda row: mindegree_dictionary[row['s']], axis=1)
# df['mindegree_o'] = df.apply(lambda row: mindegree_dictionary[row['o']], axis=1)
# df['min_mindegree'] = df.apply(lambda row: min(row['mindegree_s'], row['mindegree_o']), axis=1)
# df['mean_mindegree'] = df.apply(lambda row: (row['mindegree_s'] + row['mindegree_o'])/2, axis=1)
# df['max_mindegree'] = df.apply(lambda row: max(row['mindegree_s'], row['mindegree_o']), axis=1)

# Add real degree column
for key in df['s'].unique():
    if real_degree_dictionary[key]['in_degree'] == 0:
        real_degree_dictionary[key]['in_degree'] = 1
df['indegree_s'] = df.apply(lambda row: pow(10, int(math.log10(real_degree_dictionary[row['s']]['in_degree']) * 10) / 10), axis=1)
df['indegree_o'] = df.apply(lambda row: pow(10, int(math.log10(real_degree_dictionary[row['o']]['in_degree']) * 10) / 10), axis=1)
df['outdegree_s'] = df.apply(lambda row: pow(10, int(math.log10(real_degree_dictionary[row['s']]['out_degree']) * 10) / 10), axis=1)
df['outdegree_o'] = df.apply(lambda row: real_degree_dictionary[row['o']]['out_degree'], axis=1)
df['mean_indegree'] = df.apply(lambda row: (row['indegree_s'] + row['indegree_o'])/2, axis=1)
df['mean_outdegree'] = df.apply(lambda row: (row['outdegree_s'] + row['outdegree_o'])/2, axis=1)
df['max_indegree'] = df.apply(lambda row: max(row['indegree_s'], row['indegree_o']), axis=1)
df['max_outdegree'] = df.apply(lambda row: max(row['outdegree_s'], row['outdegree_o']), axis=1)


# # Change bool to int
df['c_original'] = df['c_original'].astype(int)
df['c_all'] = df['c_all'].astype(int)

# # groupby mindegree
df = df.groupby('indegree_o').mean()
df['relative_effect'] = df['c_all'] / df['c_original']
df.plot(y='relative_effect', title='Relative Effect of Mindegree(min)', logx=True)
print(df)
plt.show()