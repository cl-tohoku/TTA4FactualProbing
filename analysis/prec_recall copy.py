import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt



def get_variance_dict(idx, prompts_df):
    prompts_df = prompts_df.loc[prompts_df['fact_id'].isin(idx)]  

    variations = [len(prompts_df.loc[prompts_df['fact_id'] == id].groupby('result').count().index) for id in idx]
    vdict = {}
    for v in variations:
        if v in vdict.keys():
            vdict[v] += 1
        else:
            vdict[v] = 1

    return vdict


def main(folder):
    aug_path = 'cache/{}/augmentations.csv'.format(folder)
    filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)
    with open(filepath_before, 'rb') as f:
        df_before = pickle.load(f)
        df_before['confidence'] = df_before['score'] / df_before['score_total']
        df_before = df_before[['confidence', 'c_original']]
        df_before = df_before.rename(columns = {'confidence': 'conf_before', 'c_original': 'c_before'})
    with open(filepath_after, 'rb') as f:
        df_after = pickle.load(f)
        df_after['confidence'] = df_after['score'] / df_after['score_total']
        df_after = df_after[['confidence', 'c_all']]
        df_after = df_after.rename(columns = {'confidence': 'conf_after', 'c_all': 'c_after'})

    df = pd.concat([df_before,df_after], axis=1)
    print(df)

    precisions = []
    recalls = []
    for barrier in np.arange(0.0, 1.0, 0.01):
        # df_tmp = df.loc[(df['conf_before']>=barrier) & (df['conf_after']>=barrier)]
        df_tmp = df.loc[(df['conf_after']>=barrier)]
        if len(df_tmp.index) != 0:
            print(df_tmp)
            tp = df_tmp.loc[(df['c_before']==True) & (df['c_after']==True) ].count()[0]
            fn = df_tmp.loc[(df['c_before']==True) & (df['c_after']==False) ].count()[0]
            fp = df_tmp.loc[(df['c_before']==False) & (df['c_after']==True) ].count()[0]
            # tn = df_tmp.loc[(df['c_before']==False) & (df['c_after']==False) ].count()[0]
            precisions.append(tp / (tp + fp))
            recalls.append( tp / (tp + fn))
            print('precision: {}\t, recall: {}'.format(precisions[-1], recalls[-1]))

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.15, 0.20, 0.7, 0.75])
    ax.plot(recalls, precisions)
    ax.set_xlabel('Recall', fontsize = 20)
    # ax.set_xlim(0, 1)
    ax.set_ylabel('Precision', fontsize = 20)
    # ax.legend(fontsize = 18)
    fig.savefig('analysis/img/prec_recall.png')


    

if __name__ == '__main__':
    # print('---t5-small---')
    # main('v2.11d.5')
    print('\n---t5-11b---')
    main('v2.11d.4')