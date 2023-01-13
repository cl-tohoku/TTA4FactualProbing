import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    folder = 'v2.11d.4'
    filepath_before = 'cache/{}/evaluate_original.pkl'.format(folder)
    filepath_after = 'cache/{}/evaluate_all.pkl'.format(folder)

    with open(filepath_before, 'rb') as f:
        df_before = pickle.load(f)
    with open(filepath_after, 'rb') as f:
        df_after = pickle.load(f)

    df_before['confidence'] = df_before['score'] / df_before['score_total']
    df_after['confidence'] = df_after['score'] / df_after['score_total']
    print('----before-----')
    print(df_before.groupby('c_original').mean(numeric_only = True))
    print('-------------\n')
    print('----after-----')
    print(df_after.groupby('c_all').mean(numeric_only = True))
    print('-------------\n')

    plt.figure(figsize=(8, 4))
    sns.set_palette("pastel")
    sns_plot = sns.kdeplot(df_before.loc[df_before['c_original']==True]['confidence'], label='Correct (w/out TTA)',color=sns.xkcd_rgb["medium blue"], linestyle='--')
    sns_plot = sns.kdeplot(df_before.loc[df_before['c_original']==False]['confidence'], label = 'Incorrect (w/out TTA)',color=sns.xkcd_rgb["medium pink"], linestyle='--')
    # conf_max = max(df_before.loc[df_before['c_original']==True]['confidence'].max(), df_before.loc[df_before['c_original']==False]['confidence'].max())
    sns_plot = sns.kdeplot(df_after.loc[df_after['c_all']==True]['confidence'], color=sns.xkcd_rgb["medium blue"],label='Correct (w/TTA)')
    sns_plot = sns.kdeplot(df_after.loc[df_after['c_all']==False]['confidence'], color=sns.xkcd_rgb["medium pink"],label = 'Incorrect (w/TTA)')
    # conf_max = max(df_after.loc[df_after['c_all']==True]['confidence'].max(), df_after.loc[df_after['c_all']==False]['confidence'].max())
    pos = sns_plot.get_position()
    new_pos = [pos.x0, pos.y0+0.07, pos.width, pos.height]
    sns_plot.set_position(new_pos)
    sns_plot.set_xlabel('Confidence', fontsize = 20)
    # sns_plot.set_xlim(0, conf_max)
    sns_plot.set_xlim(0, 1)
    sns_plot.set_ylabel('Density', fontsize = 20)
    plt.legend(fontsize = 18)
    plt.savefig('analysis/img/confidence_11b.png')


if __name__ == '__main__':
    main()

