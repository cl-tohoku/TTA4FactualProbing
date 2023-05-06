import pandas as pd
import matplotlib.pyplot as plt

folder2name = {
        'v2.11d.gpt3.1': 't5-small.gpt3',
        'v2.11d.gpt3.2': 't5-large.gpt3',
        'v2.11d.gpt3.3': 'flan-small.gpt3',
        'v2.11d.gpt3.4': 'flan-large.gpt3',
}

def main(folder):
    name = folder2name[folder]
    dir_path = f'cache/{folder}/'
    summ_df = pd.read_table(dir_path+'augmentation_summary.csv', index_col=0).T
    summ_df['rel_dif'] = summ_df.apply(lambda row: (row['c_all']+1)/(row['c_original']+1), axis=1)
    summ_df['rel_dif'].plot()
    plt.savefig('test.png')
    plt.xticks(range(25))

if __name__ == '__main__':
    main(folder = 'v2.11d.gpt3.1')