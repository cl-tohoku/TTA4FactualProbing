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
    folder = 'v2.11d.2'
    filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)

    with open(filepath_before, 'rb') as f:
        df_before = pickle.load(f)
    

    df_before['confidence'] = df_before['score'] / df_before['score_total']
    df_before['rank'] = df_before["confidence"].rank(ascending=False, method='min')
    df_before['rank_bin'] = pd.cut(df_before['rank'], np.arange(0, 12600, 50))
    pattern = re.compile(r'^\((.*?)\,.*')
    df_before['rank_bin'] = df_before.apply(lambda row: 50 + int(pattern.match(str(row['rank_bin'])).group(1)), axis=1)
    df_before_correct = df_before.loc[df_before['c_original']==1.0].groupby('rank_bin').count()[['rank']]
    df_before_correct = df_before_correct.rolling(10).mean().shift(-10)
    df_before_incorrect = df_before.loc[df_before['c_original']==0.0].groupby('rank_bin').count()[['rank']]
    df_before_incorrect = df_before_incorrect.rolling(10).mean().shift(-10)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.75])
    ax.plot(df_before_correct.index, df_before_correct['rank'],  label='Correct (w/out TTA)',color=sns.xkcd_rgb["medium blue"], linestyle='--')
    ax.plot(df_before_incorrect.index, df_before_incorrect['rank'], label='Incorrect (w/out TTA)',color=sns.xkcd_rgb["medium pink"], linestyle='--')
    print(df_before_correct)

    with open(filepath_after, 'rb') as f:
        df_after = pickle.load(f)
    df_after['confidence'] = df_after['score'] / df_after['score_total']
    df_after['rank'] = df_after["confidence"].rank(ascending=False, method='min')
    df_after['rank_bin'] = pd.cut(df_after['rank'], np.arange(0, 12600, 50))
    df_after['rank_bin'] = df_after.apply(lambda row: 50 + int(pattern.match(str(row['rank_bin'])).group(1)), axis=1)
    df_after_correct = df_after.loc[df_after['c_all']==1.0].groupby('rank_bin').count()[['rank']]
    df_after_correct = df_after_correct.rolling(20).mean().shift(-20)
    df_after_incorrect = df_after.loc[df_after['c_all']==0.0].groupby('rank_bin').count()[['rank']]
    df_after_incorrect = df_after_incorrect.rolling(20).mean().shift(-20)

    ax.plot(df_after_correct.index, df_after_correct['rank'],  label='Correct (w/ TTA)',color=sns.xkcd_rgb["medium blue"])
    ax.plot(df_after_incorrect.index, df_after_incorrect['rank'], label='Incorrect (w/ TTA)',color=sns.xkcd_rgb["medium pink"])
    ax.set_xlabel('Rank of confidence', fontsize=15)
    ax.set_ylabel('Counts', fontsize=15)
    ax.legend()
    fig.savefig('analysis/img/confidence_rank.png')
    return 0




if __name__ == '__main__':
    main()

