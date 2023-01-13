import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

def bin_rank(df):
    df['rank_bin'] = pd.cut(df['rank'], np.arange(0, 12600, 100))
    pattern = re.compile(r'^\((.*?)\,.*')
    df['rank_bin'] = df.apply(lambda row: 50 + int(pattern.match(str(row['rank_bin'])).group(1)), axis=1)
    df_correct = df.loc[df]
    df = df.groupby('rank_bin').count()[['rank']]
    print(df)
    
#     return df


def main():
    folder = 'v2.11d.4'
    filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)

    with open(filepath_before, 'rb') as f:
        df_before = pickle.load(f)

    df_before['confidence'] = df_before['score'] / df_before['score_total']
    df_before['rank'] = df_before["confidence"].rank(ascending=False, method='min')
    df_before['rank_bin'] = pd.cut(df_before['rank'], np.arange(0, 12600, 50))
    pattern = re.compile(r'^\((.*?)\,.*')
    df_before['rank_bin'] = df_before.apply(lambda row: 50 + int(pattern.match(str(row['rank_bin'])).group(1)), axis=1)
    df_before = df_before.groupby(['rank_bin', 'c_original']).count()['rank']

    df_before_sum = pd.DataFrame(columns=['correct', 'incorrect'])
    for bin in df_before.index:
        b, j = bin
        if j:
            df_before_sum.loc[b,'correct' ] = df_before.loc[bin]
        else:
            df_before_sum.loc[b,'incorrect' ] = df_before.loc[bin]

    df_before_sum = df_before_sum.fillna(0)

    for i in range(len(df_before_sum.index)):
        if i != 0:
            df_before_sum.iloc[i] = df_before_sum.iloc[i] + df_before_sum.iloc[i-1]
    n_correct = df_before_sum.iloc[len(df_before_sum.index) - 1]['correct']

    df_before_sum['sum'] = df_before_sum.apply(lambda row: row['correct'] + row['incorrect'], axis=1)
    df_before_sum['precision'] = df_before_sum.apply(lambda row: row['correct'] / row['sum'], axis=1)
    df_before_sum['recall'] = df_before_sum.apply(lambda row: row['correct']/n_correct, axis=1)
    print(df_before_sum)

    with open(filepath_after, 'rb') as f:
        df_after = pickle.load(f)

    df_after['confidence'] = df_after['score'] / df_after['score_total']
    df_after['rank'] = df_after["confidence"].rank(ascending=False, method='min')
    df_after['rank_bin'] = pd.cut(df_after['rank'], np.arange(0, 12600, 50))
    pattern = re.compile(r'^\((.*?)\,.*')
    df_after['rank_bin'] = df_after.apply(lambda row: 50 + int(pattern.match(str(row['rank_bin'])).group(1)), axis=1)
    df_after = df_after.groupby(['rank_bin', 'c_all']).count()['rank']

    df_after_sum = pd.DataFrame(columns=['correct', 'incorrect'])
    for bin in df_after.index:
        b, j = bin
        if j:
            df_after_sum.loc[b,'correct' ] = df_after.loc[bin]
        else:
            df_after_sum.loc[b,'incorrect' ] = df_after.loc[bin]

    df_after_sum = df_after_sum.fillna(0)

    for i in range(len(df_after_sum.index)):
        if i != 0:
            df_after_sum.iloc[i] = df_after_sum.iloc[i] + df_after_sum.iloc[i-1]
    n_correct = df_after_sum.iloc[len(df_after_sum.index) - 1]['correct']

    df_after_sum['sum'] = df_after_sum.apply(lambda row: row['correct'] + row['incorrect'], axis=1)
    df_after_sum['precision'] = df_after_sum.apply(lambda row: row['correct'] / row['sum'], axis=1)
    df_after_sum['recall'] = df_after_sum.apply(lambda row: row['correct']/n_correct, axis=1)
    print(df_after_sum)


    # df_before_correct = df_before.loc[df_before['c_original']==1.0].groupby('rank_bin').count()[['rank']]
    # df_before_incorrect = df_before.loc[df_before['c_original']==0.0].groupby('rank_bin').count()[['rank']]


    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.75])
    ax.plot(df_before_sum['recall'], df_before_sum['precision'],  label='w/out TTA',color=sns.xkcd_rgb["medium blue"], linestyle='--')
    ax.plot(df_after_sum['recall'], df_after_sum['precision'],  label='w/ TTA',color=sns.xkcd_rgb["medium blue"])




    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)
    ax.legend(fontsize = 15)
    fig.savefig('analysis/img/prec_recall.pdf')
    return 0




if __name__ == '__main__':
    main()

