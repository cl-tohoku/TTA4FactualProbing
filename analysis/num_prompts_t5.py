import pandas as pd
import os
import matplotlib.pyplot as plt

from folder2label import folder2label

folders = [
    'v2.11d.5',
    'v2.11d.2',
    'v2.11d.3',
    'v2.11d.4',
    # 'v2.11d.5.cnt',
    # 'v2.11d.2.cnt',
    # 'v2.11d.3.cnt',
    # 'v2.11d.4.cnt',
    # 'v2.11d.flan-small.uncased',
    # 'v2.11d.flan.uncased',
    # 'v2.11d.t03b',
    'v2.11d.gpt3.1',
    'v2.11d.gpt3.2',
    'v2.11d.gpt3.5',
    'v2.11d.gpt3.6',
]

colors = [
    '#1f77b4', 
    '#ff7f0e', 
    '#2ca02c', 
    '#d62728',
    # '#9467bd',
    '#1f77b4', 
    '#ff7f0e', 
    '#2ca02c', 
    '#d62728',
    # '#9467bd',
]

styles = [
    'dashed',
    'dashed',
    'dashed',
    'dashed',
    # 'dashed',
    'solid',
    'solid',
    'solid',
    'solid',
    # 'solid',
]

path_pre = 'cache'

fig = plt.figure(figsize = (10, 4.1))
ax1 = fig.add_axes([0.10, 0.17, 0.56, 0.81])

for folder, color, style in zip(folders, colors, styles):
    print(folder2label(folder))
    df : pd.DataFrame
    df = pd.read_table(os.path.join('cache', folder, 'augmentation_summary.csv'), index_col=0)
    print(df)
    df = df.loc[df.index.str.contains('rand')]
    df.index = df.apply(lambda row: row.name[6:], axis=1)
    df['num_prompt'] = df.apply(lambda row: int(row.name[:-2]) if '_' in row.name else int(row.name), axis=1)
    df = df.groupby('num_prompt').mean()
    # df = df.set_index('num_prompt', drop=True)
    # print(df)
    if style == 'solid':
        df = df.loc[:, ~df.columns.isin(['P155'])]
    df['total'] = df.sum(axis=1)
    df['acc'] = df.apply(lambda row: row['total'] / df.loc[1, 'total'], axis = 1)
    # print(df)

    x = df.index.to_list()
    y = df['acc']
    ax1.plot(x, y, linestyle  = style , color= color, label = folder2label(folder))

ax1.plot([0, 31], [1, 1], "k--", linewidth = 1)
ax1.set_xlim(1, 11)
ax1.set_xlabel("Number of prompts",size=28)
ax1.set_ylabel("Relative Effect",size=28)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, framealpha=0.3, fontsize=20)
# plt.show()
dest = "analysis/img/n_prompts/t5_gpt.pdf"
plt.savefig(dest, transparent=True)

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']

print(colors)