import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

from folder2label import folder2label

def get_confidence(df):
    df['confidence'] = df['score'] / df['score_total']
    # normalize confidence
    df['confidence'] = df['confidence']  / df['confidence'].max()
    return df

def get_binned_confidence_rank(df, step):
    df['conf_bin'] = pd.cut(df['confidence'], np.arange(0, 1+step, step))
    pattern = re.compile(r'^\((.*?)\,.*')
    df['conf_bin'] = df.apply(lambda row: step + float(pattern.match(str(row['conf_bin'])).group(1)), axis=1)
    return df

def get_precision_confidence(df):
    df_sum = pd.DataFrame(columns=['correct', 'incorrect'])
    for bin in df.index:
        b, j = bin
        if j:
            df_sum.loc[b,'correct' ] = df.loc[bin]
        else:
            df_sum.loc[b,'incorrect' ] = df.loc[bin]

    df_sum = df_sum.fillna(0)
    return df_sum

def calculate(df, column):
    step = 0.05
    df = get_confidence(df)
    df = get_binned_confidence_rank(df, step)
    df = df.groupby(['conf_bin', column]).count()['os'].unstack(fill_value=0).rename(columns={True:'correct', False:'incorrect'})
    df['top_p'] = df.apply(lambda row: row.name, axis=1)
    df['accuracy'] = df.apply(lambda row: row['correct'] / (row['correct'] + row['incorrect']), axis=1)
    df['rsquare'] = df.apply(lambda row: (row['accuracy'] - row['top_p'])**2, axis=1)
    return df

def single_graph(folder, ax):
    filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)
    

    with open(filepath_before, 'rb') as f:
        df_before = pickle.load(f)
    df_before = calculate(df_before, 'c_original')


    with open(filepath_after, 'rb') as f:
        df_after = pickle.load(f)
    
    df_after = calculate(df_after, 'c_all')

    ax.plot(df_after['top_p'], df_after['accuracy'],  label='w/ TTA')
    ax.plot(df_before['top_p'], df_before['accuracy'],  label='w/out TTA')
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle=':', color='black', label=None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(labelsize= 15)
    ax.set_xlabel('Confidence', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.legend(fontsize = 15, framealpha=0.3)
    ax.set_title(folder2label(folder), fontsize=15)

def main():
    # folder = 'v2.11d.4'
    # filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    # filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)
    

    # with open(filepath_before, 'rb') as f:
    #     df_before = pickle.load(f)
    # df_before = calculate(df_before, 'c_original')


    # with open(filepath_after, 'rb') as f:
    #     df_after = pickle.load(f)
    
    # df_after = calculate(df_after, 'c_all')
    # print(df_after)

    fig, axes = plt.subplots(2, 2, tight_layout=True, figsize=(10, 6))
    single_graph('v2.11d.5', axes[0,0])
    single_graph('v2.11d.2', axes[0,1])
    single_graph('v2.11d.3', axes[1,0])
    single_graph('v2.11d.4', axes[1,1])
    # ax = fig.add_axes([0.17, 0.24, 0.78, 0.7])
    # color=sns.xkcd_rgb["medium blue"]
    # ax.plot(df_after['top_p'], df_after['accuracy'],  label='w/ TTA')
    # ax.plot(df_before['top_p'], df_before['accuracy'],  label='w/out TTA')
    # ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle=':', color='black', label=None)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.tick_params(labelsize= 20)
    # ax.set_xlabel('Normalized Confidence', fontsize=20)
    # ax.set_ylabel('Accuracy', fontsize=20)
    # ax.legend(fontsize = 15, framealpha=0.3)
    # print(folder2label(folder))
    # plt.figtext(0.55, 0.02, "High <- Confidence -> Low", ha="center", fontsize=16)
    plt.savefig('analysis/img/prec_conf/{}.pdf'.format('t5s'), transparent=True)
    return 0




if __name__ == '__main__':
    main()

